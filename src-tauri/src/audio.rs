use crate::{
    models::{AppState, Transcript},
    platform,
    storage::load_settings,
    transcription::{transcribe_audio, TranscriptionAudio},
    webm_opus::{encode_wav_as_webm_opus, read_wav_as_whisper_pcm},
};
use chrono::Utc;
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    FromSample, Sample, SampleFormat, SizedSample, SupportedStreamConfig,
};
use serde::Serialize;
use std::{
    fs,
    fs::File,
    io::BufWriter,
    path::PathBuf,
    str::FromStr,
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc, Mutex,
    },
};
use tauri::{AppHandle, State};

type WavWriter = hound::WavWriter<BufWriter<File>>;
type WavWriterHandle = Arc<Mutex<Option<WavWriter>>>;
type RecordingLevel = Arc<AtomicU32>;
type RecordingError = Arc<Mutex<Option<String>>>;

const SILENCE_THRESHOLD: f32 = 0.008;
const SILENCE_PADDING_MS: u32 = 120;

fn resolve_output_action(requested: Option<String>, fallback: &str) -> String {
    requested
        .filter(|action| matches!(action.as_str(), "paste" | "copy"))
        .unwrap_or_else(|| fallback.to_string())
}

fn microphone_value(device: &cpal::Device) -> String {
    device
        .id()
        .map(|id| id.to_string())
        .unwrap_or_else(|_| device.to_string())
}

fn configured_input_device(host: &cpal::Host, configured: &str) -> Option<cpal::Device> {
    cpal::DeviceId::from_str(configured)
        .ok()
        .and_then(|id| host.device_by_id(&id))
        .or_else(|| {
            host.input_devices()
                .ok()?
                .find(|device| device.to_string() == configured)
        })
}

pub(crate) struct NativeRecording {
    stream: cpal::Stream,
    writer: WavWriterHandle,
    path: PathBuf,
}

// CPAL's platform wrapper is conservatively non-Send even though the native
// streams are managed by the application state for the supported desktop targets.
unsafe impl Send for NativeRecording {}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct AudioDevice {
    value: String,
    label: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RecordingStatus {
    level: u32,
    error: Option<String>,
}

fn wav_spec_from_config(config: &SupportedStreamConfig) -> hound::WavSpec {
    hound::WavSpec {
        channels: config.channels(),
        sample_rate: config.sample_rate(),
        bits_per_sample: (config.sample_format().sample_size() * 8) as u16,
        sample_format: if config.sample_format().is_float() {
            hound::SampleFormat::Float
        } else {
            hound::SampleFormat::Int
        },
    }
}

fn write_input_data<T, U>(input: &[T], writer: &WavWriterHandle, level: &RecordingLevel)
where
    T: Sample,
    U: Sample + hound::Sample + FromSample<T>,
    f32: FromSample<T>,
{
    let peak = input
        .iter()
        .map(|sample| f32::from_sample(*sample).abs())
        .fold(0.0_f32, f32::max)
        .clamp(0.0, 1.0);
    let previous = level.load(Ordering::Relaxed);
    let next = (peak * 1000.0) as u32;
    level.store((previous.saturating_mul(2) + next) / 3, Ordering::Relaxed);
    if let Ok(mut guard) = writer.try_lock() {
        if let Some(writer) = guard.as_mut() {
            for &sample in input {
                let _ = writer.write_sample(U::from_sample(sample));
            }
        }
    }
}

fn record_stream_error(error: &RecordingError, stream_error: impl std::fmt::Display) {
    if let Ok(mut latest_error) = error.lock() {
        *latest_error = Some(format!(
            "Microphone connection was lost: {stream_error}. Reconnect it or choose another microphone in Settings."
        ));
    }
}

fn build_input_stream<T, U>(
    device: &cpal::Device,
    config: &SupportedStreamConfig,
    writer: WavWriterHandle,
    level: RecordingLevel,
    error: RecordingError,
) -> Result<cpal::Stream, String>
where
    T: Sample + SizedSample,
    U: Sample + hound::Sample + FromSample<T>,
    f32: FromSample<T>,
{
    let stream_config = config.clone().into();
    device
        .build_input_stream(
            stream_config,
            move |data, _| write_input_data::<T, U>(data, &writer, &level),
            move |stream_error| record_stream_error(&error, stream_error),
            None,
        )
        .map_err(|err| err.to_string())
}

fn input_stream(
    device: &cpal::Device,
    config: &SupportedStreamConfig,
    writer: WavWriterHandle,
    level: RecordingLevel,
    error: RecordingError,
) -> Result<cpal::Stream, String> {
    match config.sample_format() {
        SampleFormat::I8 => build_input_stream::<i8, i8>(device, config, writer, level, error),
        SampleFormat::I16 => build_input_stream::<i16, i16>(device, config, writer, level, error),
        SampleFormat::I32 => build_input_stream::<i32, i32>(device, config, writer, level, error),
        SampleFormat::F32 => build_input_stream::<f32, f32>(device, config, writer, level, error),
        format => Err(format!("Unsupported microphone sample format: {format}")),
    }
}

fn trim_samples<T>(
    path: &PathBuf,
    spec: hound::WavSpec,
    normalize: impl Fn(T) -> f32,
) -> Result<Option<PathBuf>, String>
where
    T: hound::Sample + Copy,
{
    let samples = hound::WavReader::open(path)
        .map_err(|err| err.to_string())?
        .into_samples::<T>()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| err.to_string())?;
    let first_sound = samples
        .iter()
        .position(|sample| normalize(*sample).abs() >= SILENCE_THRESHOLD)
        .ok_or_else(|| "No speech was detected — the microphone input was silent.".to_string())?;
    let last_sound = samples
        .iter()
        .rposition(|sample| normalize(*sample).abs() >= SILENCE_THRESHOLD)
        .expect("first_sound guarantees a final audible sample");
    let channels = usize::from(spec.channels.max(1));
    let padding = (usize::try_from(spec.sample_rate).unwrap_or_default()
        * usize::from(spec.channels)
        * usize::try_from(SILENCE_PADDING_MS).unwrap_or_default())
        / 1_000;
    let start = (first_sound.saturating_sub(padding) / channels) * channels;
    let end = ((last_sound + 1 + padding).min(samples.len()) / channels) * channels;

    if start == 0 && end == samples.len() {
        return Ok(None);
    }

    let trimmed_path = path.with_file_name(format!(
        "{}-trimmed.wav",
        path.file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("capture")
    ));
    let mut writer =
        hound::WavWriter::create(&trimmed_path, spec).map_err(|err| err.to_string())?;
    for sample in &samples[start..end] {
        writer
            .write_sample(*sample)
            .map_err(|err| err.to_string())?;
    }
    writer.finalize().map_err(|err| err.to_string())?;
    Ok(Some(trimmed_path))
}

fn trim_silence(path: &PathBuf) -> Result<Option<PathBuf>, String> {
    let reader = hound::WavReader::open(path).map_err(|err| err.to_string())?;
    let spec = reader.spec();
    drop(reader);

    match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Float, _) => trim_samples::<f32>(path, spec, |sample| sample),
        (hound::SampleFormat::Int, 8) => {
            trim_samples::<i8>(path, spec, |sample| sample as f32 / i8::MAX as f32)
        }
        (hound::SampleFormat::Int, 16) => {
            trim_samples::<i16>(path, spec, |sample| sample as f32 / i16::MAX as f32)
        }
        (hound::SampleFormat::Int, _) => {
            trim_samples::<i32>(path, spec, |sample| sample as f32 / i32::MAX as f32)
        }
    }
}

pub(crate) fn get_microphones() -> Result<Vec<AudioDevice>, String> {
    let host = cpal::default_host();
    let mut devices = vec![AudioDevice {
        value: "Default microphone".into(),
        label: "Default microphone".into(),
    }];
    for device in host.input_devices().map_err(|err| {
        format!(
            "Couldn’t enumerate microphones: {err}. {}",
            platform::microphone_permission_hint()
        )
    })? {
        let name = device.to_string();
        devices.push(AudioDevice {
            value: microphone_value(&device),
            label: name,
        });
    }
    Ok(devices)
}

pub(crate) fn get_recording_status(state: State<AppState>) -> RecordingStatus {
    RecordingStatus {
        level: state.recording_level.load(Ordering::Relaxed),
        error: state
            .recording_error
            .lock()
            .ok()
            .and_then(|error| error.clone()),
    }
}

pub(crate) fn start_native_recording(state: State<AppState>) -> Result<(), String> {
    let mut active = state
        .recording
        .lock()
        .map_err(|_| "Recording is unavailable".to_string())?;
    if active.is_some() {
        return Ok(());
    }
    state.recording_level.store(0, Ordering::Relaxed);
    if let Ok(mut error) = state.recording_error.lock() {
        *error = None;
    }
    let settings = load_settings(&state);
    let host = cpal::default_host();
    let device = if settings.microphone == "Default microphone" {
        host.default_input_device().ok_or_else(|| {
            format!(
                "No microphone is available. Connect one and make sure desktop apps can use it. {}",
                platform::microphone_permission_hint()
            )
        })?
    } else {
        configured_input_device(&host, &settings.microphone)
            .ok_or_else(|| {
                format!(
                    "The selected microphone “{}” is unavailable. Reconnect it or choose another microphone in Settings.",
                    settings.microphone
                )
            })?
    };
    let config = device.default_input_config().map_err(|err| {
        format!(
            "Couldn’t open the microphone: {err}. {}",
            platform::microphone_permission_hint()
        )
    })?;
    let path = state
        .data_dir
        .join(format!("capture-{}.wav", Utc::now().timestamp_millis()));
    let writer = hound::WavWriter::create(&path, wav_spec_from_config(&config))
        .map_err(|err| err.to_string())?;
    let writer = Arc::new(Mutex::new(Some(writer)));
    let stream = input_stream(
        &device,
        &config,
        writer.clone(),
        state.recording_level.clone(),
        state.recording_error.clone(),
    )?;
    stream.play().map_err(|err| {
        format!("Couldn’t start the microphone: {err}. Another app may be using it exclusively.")
    })?;
    *active = Some(NativeRecording {
        stream,
        writer,
        path,
    });
    Ok(())
}

pub(crate) fn cancel_native_recording(state: State<AppState>) -> Result<(), String> {
    let recording = state
        .recording
        .lock()
        .map_err(|_| "Recording is unavailable".to_string())?
        .take();
    state.recording_level.store(0, Ordering::Relaxed);
    if let Some(recording) = recording {
        drop(recording.stream);
        if let Ok(mut writer) = recording.writer.lock() {
            if let Some(writer) = writer.take() {
                let _ = writer.finalize();
            }
        }
        let _ = fs::remove_file(recording.path);
    }
    Ok(())
}

pub(crate) async fn stop_native_recording(
    app: AppHandle,
    state: State<'_, AppState>,
    output_action: Option<String>,
) -> Result<Transcript, String> {
    let recording = state
        .recording
        .lock()
        .map_err(|_| "Recording is unavailable".to_string())?
        .take()
        .ok_or_else(|| "There isn’t an active recording.".to_string())?;
    let recording_error = state
        .recording_error
        .lock()
        .map_err(|_| "Recording is unavailable".to_string())?
        .take();
    state.recording_level.store(0, Ordering::Relaxed);
    drop(recording.stream);
    recording
        .writer
        .lock()
        .map_err(|_| "Couldn’t finalize the recording.".to_string())?
        .take()
        .ok_or_else(|| "Couldn’t finalize the recording.".to_string())?
        .finalize()
        .map_err(|err| err.to_string())?;
    if let Some(error) = recording_error {
        let _ = fs::remove_file(&recording.path);
        return Err(error);
    }
    let trimmed_path = match trim_silence(&recording.path) {
        Ok(path) => path,
        Err(error) => {
            let _ = fs::remove_file(&recording.path);
            return Err(error);
        }
    };
    let audio_path = trimmed_path.as_ref().unwrap_or(&recording.path);
    let settings = load_settings(&state);
    let encoded_audio = match settings.transcription_provider.as_str() {
        "local" => read_wav_as_whisper_pcm(audio_path).map(TranscriptionAudio::WhisperPcm),
        _ => encode_wav_as_webm_opus(audio_path).map(TranscriptionAudio::WebmOpus),
    };
    let audio = match encoded_audio {
        Ok(audio) => audio,
        Err(error) => {
            let _ = fs::remove_file(&recording.path);
            if let Some(trimmed_path) = trimmed_path {
                let _ = fs::remove_file(trimmed_path);
            }
            return Err(error);
        }
    };
    let _ = fs::remove_file(&recording.path);
    if let Some(trimmed_path) = trimmed_path {
        let _ = fs::remove_file(trimmed_path);
    }
    let output_action = resolve_output_action(output_action, &settings.output_action);
    let result = transcribe_audio(app, &state, audio.clone(), &output_action).await;
    if let Ok(mut failed_audio) = state.last_failed_audio.lock() {
        *failed_audio = result.as_ref().err().map(|_| audio);
    }
    result
}

pub(crate) async fn retry_last_transcription(
    app: AppHandle,
    state: State<'_, AppState>,
    output_action: Option<String>,
) -> Result<Transcript, String> {
    let audio = state
        .last_failed_audio
        .lock()
        .map_err(|_| "Retry is unavailable".to_string())?
        .clone()
        .ok_or_else(|| "There is no failed dictation to retry.".to_string())?;
    let settings = load_settings(&state);
    let output_action = resolve_output_action(output_action, &settings.output_action);
    let result = transcribe_audio(app, &state, audio, &output_action).await;
    if result.is_ok() {
        if let Ok(mut failed_audio) = state.last_failed_audio.lock() {
            *failed_audio = None;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::resolve_output_action;

    #[test]
    fn retry_output_action_accepts_only_supported_values() {
        assert_eq!(resolve_output_action(Some("copy".into()), "paste"), "copy");
        assert_eq!(resolve_output_action(Some("paste".into()), "copy"), "paste");
        assert_eq!(resolve_output_action(Some("type".into()), "copy"), "copy");
        assert_eq!(resolve_output_action(None, "paste"), "paste");
    }
}
