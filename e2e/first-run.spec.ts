import { expect, test } from "@playwright/test";

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    const settings = {
      hotkey: "Ctrl+Shift+Space",
      inputMode: "hold",
      microphone: "Default microphone",
      model: "whisper-large-v3-turbo",
      language: "auto",
      outputAction: "paste",
      keepRunningInTray: true,
      launchAtLogin: false,
      startInTray: false,
      historyRetention: "15",
      notificationsEnabled: true,
      completedOnboarding: false,
      voiceCommandsEnabled: false,
      textMode: "literal",
      personalVocabulary: "",
      dictionaryReplacements: "",
      autoInstallUpdates: false,
      deferredUpdateVersion: "",
      transcriptionProvider: "groq",
      localWhisperModel: "base",
      localWhisperAcceleration: "auto",
      discordPushToMuteEnabled: false,
    };
    type EventCallback = (event: { payload: string }) => void;
    const callbacks = new Map<number, EventCallback>();
    const eventListeners = new Map<string, number>();
    let nextCallback = 0;
    let retryableDictation = false;
    let history: {
      id: string;
      text: string;
      createdAt: string;
      language: string;
      pinned: boolean;
    }[] = [];
    const emitShortcut = (payload: string) => {
      const callbackId = eventListeners.get("veskri-shortcut");
      callbacks.get(callbackId ?? -1)?.({ payload });
    };
    (
      window as unknown as {
        __TAURI_INTERNALS__: unknown;
        __veskriTest: { emitShortcut: (payload: string) => void };
      }
    ).__TAURI_INTERNALS__ = {
      metadata: { currentWindow: { label: "main" } },
      transformCallback: (callback: EventCallback) => {
        nextCallback += 1;
        callbacks.set(nextCallback, callback);
        return nextCallback;
      },
      unregisterCallback: () => undefined,
      unregisterListener: () => undefined,
      invoke: async (
        command: string,
        args: {
          settings?: typeof settings;
          event?: string;
          handler?: number;
        } = {},
      ) => {
        if (command === "get_settings") return settings;
        if (command === "get_history") return history;
        if (command === "has_api_key") return true;
        if (command === "get_microphones")
          return [{ value: "Default microphone", label: "Default microphone" }];
        if (command === "save_settings") {
          Object.assign(settings, args.settings);
          return settings;
        }
        if (command === "plugin:event|listen") {
          if (args.event && args.handler)
            eventListeners.set(args.event, args.handler);
          return args.handler;
        }
        if (command === "start_native_recording") return null;
        if (command === "get_recording_status")
          return { level: 0, error: null };
        if (command === "stop_native_recording") {
          retryableDictation = true;
          throw "Groq is temporarily unavailable. Your audio is ready to retry.";
        }
        if (command === "has_retryable_dictation") return retryableDictation;
        if (command === "retry_last_transcription") {
          retryableDictation = false;
          const transcript = {
            id: "retry-success",
            text: "Retried dictation",
            createdAt: "2026-09-05T18:00:00Z",
            language: "en",
            pinned: false,
          };
          history = [transcript];
          return transcript;
        }
        if (command === "plugin:updater|check") return null;
        return null;
      },
    };
    (
      window as unknown as {
        __veskriTest: { emitShortcut: (payload: string) => void };
      }
    ).__veskriTest = { emitShortcut };
    (
      window as unknown as { __TAURI_EVENT_PLUGIN_INTERNALS__: unknown }
    ).__TAURI_EVENT_PLUGIN_INTERNALS__ = {
      unregisterListener: () => undefined,
    };
  });
});

test("first-run onboarding can be completed without a key", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page.getByText("WELCOME TO VESKRI")).toBeVisible();
  await page.getByRole("button", { name: "Set up a key later" }).click();
  await expect(page.getByRole("heading", { name: "Settings" })).toBeVisible();
});

test("a temporary Groq transcription failure can retry and restore history", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Set up a key later" }).click();
  await expect(page.getByRole("heading", { name: "Settings" })).toBeVisible();

  await page.evaluate(() => window.__veskriTest.emitShortcut("pressed"));
  await expect(
    page.getByText("Listening… release when you’re done"),
  ).toBeVisible();
  await page.evaluate(() => window.__veskriTest.emitShortcut("released"));
  await expect(page.getByText("Groq is temporarily unavailable")).toBeVisible();

  await page.getByRole("button", { name: "Retry" }).click();
  await expect(page.getByText("Pasted into your active app")).toBeVisible();
  await page.getByRole("link", { name: "History" }).click();
  await expect(page.getByText("Retried dictation")).toBeVisible();
});
