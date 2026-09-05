use crate::{models::AppSettings, storage::local_models_dir};
use fs2::{available_space, FileExt};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
};
use sysinfo::System;
use whisper_rs::{
    get_lang_str, FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters,
};

#[cfg(any(
    all(feature = "local-whisper-cuda", feature = "local-whisper-rocm"),
    all(feature = "local-whisper-cuda", feature = "local-whisper-vulkan"),
    all(feature = "local-whisper-cuda", feature = "local-whisper-metal"),
    all(feature = "local-whisper-cuda", feature = "local-whisper-intel-sycl"),
    all(feature = "local-whisper-rocm", feature = "local-whisper-vulkan"),
    all(feature = "local-whisper-rocm", feature = "local-whisper-metal"),
    all(feature = "local-whisper-rocm", feature = "local-whisper-intel-sycl"),
    all(feature = "local-whisper-vulkan", feature = "local-whisper-metal"),
    all(feature = "local-whisper-vulkan", feature = "local-whisper-intel-sycl"),
    all(feature = "local-whisper-metal", feature = "local-whisper-intel-sycl")
))]
compile_error!(
    "Local Whisper GPU builds must select one backend: local-whisper-cuda, local-whisper-rocm, local-whisper-vulkan, local-whisper-metal, or local-whisper-intel-sycl."
);

#[cfg(all(feature = "local-whisper-rocm", not(target_os = "linux")))]
compile_error!("The Local Whisper ROCm backend is supported only on Linux.");

#[cfg(all(
    feature = "local-whisper-cuda",
    not(any(target_os = "linux", target_os = "windows"))
))]
compile_error!("The Local Whisper CUDA backend is supported only on Windows and Linux.");

#[cfg(all(feature = "local-whisper-metal", not(target_os = "macos")))]
compile_error!("The Local Whisper Metal backend is supported only on macOS.");

#[cfg(all(
    feature = "local-whisper-intel-sycl",
    not(any(target_os = "linux", target_os = "windows"))
))]
compile_error!("The Local Whisper Intel SYCL backend is supported only on Windows and Linux.");

#[derive(Clone, Copy)]
struct LocalModelSpec {
    id: &'static str,
    name: &'static str,
    description: &'static str,
    file_name: &'static str,
    url: &'static str,
    size_bytes: u64,
    sha256: &'static str,
    download_size_mib: u32,
    estimated_ram_mib: u32,
    estimated_vram_mib: u32,
}

const MODELS: [LocalModelSpec; 6] = [
    LocalModelSpec {
        id: "tiny",
        name: "Tiny",
        description: "Fastest local option. Best for short notes and fast CPUs.",
        file_name: "ggml-tiny.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin?download=true",
        size_bytes: 77_691_713,
        sha256: "be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21",
        download_size_mib: 75,
        estimated_ram_mib: 350,
        estimated_vram_mib: 350,
    },
    LocalModelSpec {
        id: "base",
        name: "Base",
        description: "Recommended balance of speed and accuracy for everyday dictation.",
        file_name: "ggml-base.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin?download=true",
        size_bytes: 147_951_465,
        sha256: "60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe",
        download_size_mib: 142,
        estimated_ram_mib: 550,
        estimated_vram_mib: 550,
    },
    LocalModelSpec {
        id: "small",
        name: "Small",
        description: "Higher accuracy for multilingual dictation; needs more memory.",
        file_name: "ggml-small.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin?download=true",
        size_bytes: 487_601_967,
        sha256: "1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b",
        download_size_mib: 466,
        estimated_ram_mib: 1_200,
        estimated_vram_mib: 1_200,
    },
    LocalModelSpec {
        id: "medium",
        name: "Medium",
        description: "Best local accuracy in the initial release; slower on CPU.",
        file_name: "ggml-medium.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin?download=true",
        size_bytes: 1_533_763_059,
        sha256: "6c14d5adee5f86394037b4e4e8b59f1673b6cee10e3cf0b11bbdbee79c156208",
        download_size_mib: 1_463,
        estimated_ram_mib: 3_000,
        estimated_vram_mib: 3_000,
    },
    LocalModelSpec {
        id: "large-v3-turbo",
        name: "Large v3 Turbo",
        description: "Recommended high-quality local model. Much faster than Large v3.",
        file_name: "ggml-large-v3-turbo.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin?download=true",
        size_bytes: 1_624_555_275,
        sha256: "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69",
        download_size_mib: 1_550,
        estimated_ram_mib: 3_500,
        estimated_vram_mib: 3_500,
    },
    LocalModelSpec {
        id: "large-v3",
        name: "Large v3",
        description: "Highest local multilingual accuracy. Best with a capable GPU.",
        file_name: "ggml-large-v3.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin?download=true",
        size_bytes: 3_095_033_483,
        sha256: "64d182b440b98d5203c4f9bd541544d84c605196c4f7b845dfa11fb23594d1e2",
        download_size_mib: 2_952,
        estimated_ram_mib: 6_000,
        estimated_vram_mib: 6_000,
    },
];

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct LocalModelInfo {
    id: &'static str,
    name: &'static str,
    description: &'static str,
    download_size_mib: u32,
    estimated_ram_mib: u32,
    estimated_vram_mib: u32,
    required_free_disk_mib: u32,
    installed: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct LocalWhisperCapabilities {
    cpu_available: bool,
    cuda_available: bool,
    rocm_available: bool,
    vulkan_available: bool,
    metal_available: bool,
    intel_sycl_available: bool,
    available_memory_mib: u64,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct LocalModelDownloadProgress {
    pub(crate) id: String,
    pub(crate) progress: u8,
}

struct LoadedModel {
    id: String,
    use_gpu: bool,
    context: WhisperContext,
}

static LOADED_MODEL: OnceLock<Mutex<Option<LoadedModel>>> = OnceLock::new();

fn specs(id: &str) -> Result<&'static LocalModelSpec, String> {
    MODELS
        .iter()
        .find(|model| model.id == id)
        .ok_or_else(|| "That local Whisper model is not supported.".into())
}

fn model_path(data_dir: &Path, id: &str) -> Result<PathBuf, String> {
    Ok(local_models_dir(data_dir).join(specs(id)?.file_name))
}

fn verification_path(data_dir: &Path, model: &LocalModelSpec) -> PathBuf {
    local_models_dir(data_dir).join(format!("{}.sha256", model.file_name))
}

pub(crate) fn models(data_dir: &Path) -> Vec<LocalModelInfo> {
    MODELS
        .iter()
        .map(|model| LocalModelInfo {
            id: model.id,
            name: model.name,
            description: model.description,
            download_size_mib: model.download_size_mib,
            estimated_ram_mib: model.estimated_ram_mib,
            estimated_vram_mib: model.estimated_vram_mib,
            required_free_disk_mib: bytes_to_mib(required_download_space(model)),
            installed: model_file_is_verified(data_dir, model),
        })
        .collect()
}

pub(crate) fn capabilities() -> LocalWhisperCapabilities {
    LocalWhisperCapabilities {
        cpu_available: true,
        cuda_available: cfg!(feature = "local-whisper-cuda"),
        rocm_available: cfg!(feature = "local-whisper-rocm"),
        vulkan_available: cfg!(feature = "local-whisper-vulkan"),
        metal_available: cfg!(feature = "local-whisper-metal"),
        intel_sycl_available: cfg!(feature = "local-whisper-intel-sycl"),
        available_memory_mib: available_memory_mib(),
    }
}

pub(crate) async fn download_model<F>(
    data_dir: &Path,
    id: &str,
    mut report_progress: F,
) -> Result<(), String>
where
    F: FnMut(u8),
{
    let model = specs(id)?;
    let directory = local_models_dir(data_dir);
    fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
    let destination = directory.join(model.file_name);
    if model_file_is_verified(data_dir, model) {
        return Ok(());
    }
    ensure_free_download_space(&directory, model)?;

    let lock_path = destination.with_extension("download.lock");
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(lock_path)
        .map_err(|_| "Couldn’t prepare the local model download.")?;
    lock.try_lock_exclusive().map_err(|_| {
        "This local Whisper model is already downloading. Wait for it to finish before retrying.".to_owned()
    })?;

    if model_file_has_expected_size(data_dir, model) {
        if sha256_file(&destination)? == model.sha256 {
            write_verification_stamp(data_dir, model)?;
            return Ok(());
        }
        fs::remove_file(&destination)
            .map_err(|_| "Couldn’t replace the invalid local Whisper model.")?;
    }

    let temporary = destination.with_extension("part");
    let _ = fs::remove_file(&temporary);
    report_progress(0);
    let result = async {
        let response = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(15))
            .timeout(std::time::Duration::from_secs(60 * 60))
            .build()
            .map_err(|_| "Couldn’t prepare the local Whisper download.")?
            .get(model.url)
            .send()
            .await
            .map_err(|_| {
                "Couldn’t download the local Whisper model. Check your connection and retry."
            })?
            .error_for_status()
            .map_err(|_| "The local Whisper model download was unavailable. Try again later.")?;
        if response
            .content_length()
            .is_some_and(|size| size != model.size_bytes)
        {
            return Err(
                "The local Whisper model download had an unexpected size. Try again later.".into(),
            );
        }
        let mut file =
            File::create(&temporary).map_err(|_| "Couldn’t create the local model file.")?;
        let mut response = response;
        let mut bytes_written = 0_u64;
        let mut last_progress = 0_u8;
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|_| "The local Whisper model download was interrupted.")?
        {
            file.write_all(&chunk).map_err(|_| {
                "Couldn’t save the local Whisper model. Check free disk space and retry."
            })?;
            bytes_written += chunk.len() as u64;
            if bytes_written > model.size_bytes {
                return Err(
                    "The local Whisper model download exceeded its expected size. Try again later."
                        .into(),
                );
            }
            let progress = ((bytes_written * 100) / model.size_bytes).min(99) as u8;
            if progress > last_progress {
                last_progress = progress;
                report_progress(progress);
            }
        }
        file.flush()
            .and_then(|()| file.sync_all())
            .map_err(|_| "Couldn’t finish writing the local Whisper model.")?;
        if bytes_written != model.size_bytes {
            return Err("The local Whisper model download was incomplete. Please retry.".into());
        }
        let checksum = sha256_file(&temporary)?;
        if checksum != model.sha256 {
            return Err("The local Whisper model failed its integrity check. Please retry.".into());
        }
        fs::rename(&temporary, &destination)
            .map_err(|_| "Couldn’t finalize the local Whisper model download.")?;
        write_verification_stamp(data_dir, model)?;
        report_progress(100);
        Ok(())
    }
    .await;
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

pub(crate) fn delete_model(data_dir: &Path, id: &str) -> Result<(), String> {
    let path = model_path(data_dir, id)?;
    if let Some(cache) = LOADED_MODEL.get() {
        let mut cache = cache.lock().map_err(|_| "Local Whisper is unavailable.")?;
        if cache.as_ref().is_some_and(|model| model.id == id) {
            *cache = None;
        }
    }
    if path.exists() {
        fs::remove_file(path).map_err(|error| error.to_string())?;
    }
    let model = specs(id)?;
    let _ = fs::remove_file(verification_path(data_dir, model));
    Ok(())
}

fn should_use_gpu(settings: &AppSettings) -> bool {
    match settings.local_whisper_acceleration.as_str() {
        "cuda" => cfg!(feature = "local-whisper-cuda"),
        "rocm" => cfg!(feature = "local-whisper-rocm"),
        "vulkan" => cfg!(feature = "local-whisper-vulkan"),
        "metal" => cfg!(feature = "local-whisper-metal"),
        "intel-sycl" => cfg!(feature = "local-whisper-intel-sycl"),
        "auto" => {
            cfg!(feature = "local-whisper-cuda")
                || cfg!(feature = "local-whisper-rocm")
                || cfg!(feature = "local-whisper-vulkan")
                || cfg!(feature = "local-whisper-metal")
                || cfg!(feature = "local-whisper-intel-sycl")
        }
        _ => false,
    }
}

pub(crate) fn transcribe(
    data_dir: &Path,
    settings: &AppSettings,
    samples: &[f32],
) -> Result<(String, String), String> {
    let model_path = model_path(data_dir, &settings.local_whisper_model)?;
    let model = specs(&settings.local_whisper_model)?;
    if !model_file_is_verified(data_dir, model) {
        return Err("Download the selected local Whisper model before dictating offline.".into());
    }
    ensure_memory_for_transcription(model)?;
    let use_gpu = should_use_gpu(settings);
    let cache = LOADED_MODEL.get_or_init(|| Mutex::new(None));
    let mut cache = cache.lock().map_err(|_| "Local Whisper is unavailable.")?;
    let requires_reload = cache.as_ref().is_none_or(|loaded| {
        loaded.id != settings.local_whisper_model || loaded.use_gpu != use_gpu
    });
    if requires_reload {
        let mut params = WhisperContextParameters::default();
        params.use_gpu(use_gpu);
        let context = WhisperContext::new_with_params(&model_path, params)
            .map_err(|error| format!("Couldn’t load the local Whisper model: {error}"))?;
        *cache = Some(LoadedModel {
            id: settings.local_whisper_model.clone(),
            use_gpu,
            context,
        });
    }
    let loaded = cache.as_ref().expect("model cache was initialized");
    let mut state = loaded
        .context
        .create_state()
        .map_err(|error| format!("Couldn’t prepare local Whisper: {error}"))?;
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params
        .set_n_threads(std::thread::available_parallelism().map_or(4, |count| count.get()) as i32);
    params.set_language((settings.language != "auto").then_some(settings.language.as_str()));
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_no_context(true);
    state
        .full(params, samples)
        .map_err(|error| format!("Local Whisper couldn’t transcribe this dictation: {error}"))?;
    let text = state
        .as_iter()
        .map(|segment| segment.to_string())
        .collect::<String>()
        .trim()
        .to_owned();
    let language = get_lang_str(state.full_lang_id_from_state())
        .unwrap_or("auto")
        .to_owned();
    Ok((text, language))
}

const DOWNLOAD_HEADROOM_BYTES: u64 = 64 * 1024 * 1024;

fn required_download_space(model: &LocalModelSpec) -> u64 {
    model.size_bytes + DOWNLOAD_HEADROOM_BYTES
}

fn bytes_to_mib(bytes: u64) -> u32 {
    bytes.div_ceil(1024 * 1024) as u32
}

fn model_file_has_expected_size(data_dir: &Path, model: &LocalModelSpec) -> bool {
    local_models_dir(data_dir)
        .join(model.file_name)
        .metadata()
        .is_ok_and(|metadata| metadata.is_file() && metadata.len() == model.size_bytes)
}

fn model_file_is_verified(data_dir: &Path, model: &LocalModelSpec) -> bool {
    model_file_has_expected_size(data_dir, model)
        && fs::read_to_string(verification_path(data_dir, model))
            .is_ok_and(|checksum| checksum.trim() == model.sha256)
}

fn write_verification_stamp(data_dir: &Path, model: &LocalModelSpec) -> Result<(), String> {
    fs::write(
        verification_path(data_dir, model),
        format!("{}\n", model.sha256),
    )
    .map_err(|_| "Couldn’t record the local Whisper integrity check.".to_owned())
}

fn available_memory_mib() -> u64 {
    let mut system = System::new();
    system.refresh_memory();
    system.available_memory() / (1024 * 1024)
}

fn ensure_memory_for_transcription(model: &LocalModelSpec) -> Result<(), String> {
    let available = available_memory_mib();
    if available < u64::from(model.estimated_ram_mib) {
        return Err(format!(
            "{} local Whisper needs about {} MiB of free memory, but only {} MiB is currently available. Close other apps or choose a smaller model.",
            model.name, model.estimated_ram_mib, available
        ));
    }
    Ok(())
}

fn ensure_free_download_space(directory: &Path, model: &LocalModelSpec) -> Result<(), String> {
    let available = available_space(directory)
        .map_err(|_| "Couldn’t determine available disk space for the local model download.")?;
    let required = required_download_space(model);
    if available < required {
        return Err(format!(
            "{} local Whisper needs {} MiB of free disk space to download safely, but only {} MiB is available.",
            model.name,
            bytes_to_mib(required),
            bytes_to_mib(available)
        ));
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = File::open(path).map_err(|_| "Couldn’t verify the local Whisper model.")?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|_| "Couldn’t verify the local Whisper model.")?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::{
        bytes_to_mib, capabilities, models, required_download_space, sha256_file, should_use_gpu,
        MODELS,
    };
    use crate::models::AppSettings;
    use std::{env, fs};

    #[test]
    fn exposes_multiple_optional_local_models() {
        let models = models(&env::temp_dir());
        assert_eq!(models.len(), 6);
        assert!(models.iter().any(|model| model.id == "large-v3-turbo"));
        assert!(models.iter().any(|model| model.id == "large-v3"));
        assert!(models.iter().all(|model| model.estimated_vram_mib > 0));
        assert_eq!(models[0].required_free_disk_mib, 139);
        assert_eq!(
            models
                .iter()
                .find(|model| model.id == "medium")
                .expect("medium model is exposed")
                .download_size_mib,
            1_463
        );
        assert!(MODELS
            .iter()
            .all(|model| required_download_space(model) > model.size_bytes));
        assert_eq!(bytes_to_mib(1_048_576), 1);
        assert!(capabilities().cpu_available);
        assert_eq!(
            capabilities().cuda_available,
            cfg!(feature = "local-whisper-cuda")
        );
        assert_eq!(
            capabilities().vulkan_available,
            cfg!(feature = "local-whisper-vulkan")
        );
        assert_eq!(
            capabilities().rocm_available,
            cfg!(feature = "local-whisper-rocm")
        );
        assert_eq!(
            capabilities().metal_available,
            cfg!(feature = "local-whisper-metal")
        );
        assert_eq!(
            capabilities().intel_sycl_available,
            cfg!(feature = "local-whisper-intel-sycl")
        );
    }

    #[test]
    fn selects_only_the_gpu_backend_compiled_into_this_build() {
        let mut settings = AppSettings::default();
        settings.local_whisper_acceleration = "cpu".into();
        assert!(!should_use_gpu(&settings));

        settings.local_whisper_acceleration = "auto".into();
        assert_eq!(
            should_use_gpu(&settings),
            cfg!(feature = "local-whisper-cuda")
                || cfg!(feature = "local-whisper-rocm")
                || cfg!(feature = "local-whisper-vulkan")
                || cfg!(feature = "local-whisper-metal")
                || cfg!(feature = "local-whisper-intel-sycl")
        );

        settings.local_whisper_acceleration = "cuda".into();
        assert_eq!(
            should_use_gpu(&settings),
            cfg!(feature = "local-whisper-cuda")
        );

        settings.local_whisper_acceleration = "vulkan".into();
        assert_eq!(
            should_use_gpu(&settings),
            cfg!(feature = "local-whisper-vulkan")
        );

        settings.local_whisper_acceleration = "rocm".into();
        assert_eq!(
            should_use_gpu(&settings),
            cfg!(feature = "local-whisper-rocm")
        );

        settings.local_whisper_acceleration = "metal".into();
        assert_eq!(
            should_use_gpu(&settings),
            cfg!(feature = "local-whisper-metal")
        );

        settings.local_whisper_acceleration = "intel-sycl".into();
        assert_eq!(
            should_use_gpu(&settings),
            cfg!(feature = "local-whisper-intel-sycl")
        );
    }

    #[test]
    fn hashes_downloads_with_sha256() {
        let path = env::temp_dir().join(format!(
            "veskri-local-whisper-hash-{}-{}.tmp",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock is after the Unix epoch")
                .as_nanos()
        ));
        fs::write(&path, "veskri").expect("write test model fragment");
        assert_eq!(
            sha256_file(&path).expect("hash test model fragment"),
            "a9d49a9b85d8ea1afe5e49f445fec441c2b4453006e3ec01ce3c39c4162e034e"
        );
        fs::remove_file(path).expect("remove test model fragment");
    }
}
