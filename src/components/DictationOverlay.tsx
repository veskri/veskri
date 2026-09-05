import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { Check, CircleAlert, LoaderCircle, Mic } from "lucide-react";
import type { OverlayPayload, RecordingStatus } from "../types";

export default function DictationOverlay() {
  const [overlay, setOverlay] = useState<OverlayPayload>({
    state: "listening",
    message: "Listening — release to transcribe",
  });
  const [inputLevel, setInputLevel] = useState(0);
  const [meterFrame, setMeterFrame] = useState(0);

  useEffect(() => {
    document.documentElement.classList.add("overlay-window");
    document.body.classList.add("overlay-window");
    return () => {
      document.documentElement.classList.remove("overlay-window");
      document.body.classList.remove("overlay-window");
    };
  }, []);

  useEffect(() => {
    const unlisten = listen<OverlayPayload>("veskri-overlay", (event) => {
      setOverlay(event.payload);
      if (event.payload.state !== "listening") setInputLevel(0);
    });
    return () => void unlisten.then((fn) => fn());
  }, []);

  useEffect(() => {
    if (overlay.state !== "listening") return;
    let disposed = false;
    let pollInFlight = false;
    const updateLevel = async () => {
      if (pollInFlight) return;
      pollInFlight = true;
      try {
        const status = await invoke<RecordingStatus>("get_recording_status");
        if (disposed) return;
        setInputLevel(status.level);
        setMeterFrame((frame) => frame + 1);
      } catch {
        // The main window handles recording errors; the overlay can stay quiet.
      } finally {
        pollInFlight = false;
      }
    };
    void updateLevel();
    const timer = window.setInterval(() => void updateLevel(), 100);
    return () => {
      disposed = true;
      window.clearInterval(timer);
    };
  }, [overlay.state]);

  useEffect(() => {
    if (overlay.state !== "success" && overlay.state !== "error") return;
    const timer = window.setTimeout(() => {
      void invoke("hide_dictation_overlay");
    }, 1_800);
    return () => window.clearTimeout(timer);
  }, [overlay.state]);

  const meterLevel = Math.min(Math.pow(inputLevel / 360, 0.55), 1);
  const icon =
    overlay.state === "success" ? (
      <Check size={19} />
    ) : overlay.state === "error" ? (
      <CircleAlert size={19} />
    ) : overlay.state === "transcribing" ? (
      <LoaderCircle size={19} className="spin" />
    ) : (
      <Mic size={19} />
    );

  return (
    <main
      className={`dictation-overlay ${overlay.state}`}
      role="status"
      aria-live={overlay.state === "error" ? "assertive" : "polite"}
      aria-atomic="true"
    >
      <div className="overlay-icon">{icon}</div>
      <div className="overlay-copy">
        <strong>
          {overlay.state === "listening"
            ? "Listening"
            : overlay.state === "transcribing"
              ? "Transcribing"
              : overlay.state === "success"
                ? "Complete"
                : "Needs attention"}
        </strong>
        <span>{overlay.message}</span>
      </div>
      <div className="overlay-meter" aria-hidden="true">
        {[1, 2, 3, 4, 5, 6, 7].map((bar) => {
          const motion = 10 + ((bar * 17 + meterFrame * (bar + 4)) % 24);
          return (
            <i
              key={bar}
              style={{ height: `${Math.round(5 + meterLevel * motion)}px` }}
            />
          );
        })}
      </div>
    </main>
  );
}
