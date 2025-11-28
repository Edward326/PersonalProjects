package com.visionassist.appspace.models;

import static com.visionassist.appspace.utils.UtilsKt.load_sceneClassifierError;
import static com.visionassist.appspace.utils.UtilsKt.load_speechRecognizerError;
import static com.visionassist.appspace.utils.UtilsKt.load_translaterError;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import com.visionassist.appspace.PhoneStatusMonitor;
import com.visionassist.appspace.database.NetworkUtils;
import com.visionassist.appspace.jetpack.managers.ErrorDialogManager;
import com.visionassist.appspace.jetpack.managers.InfoNotificationManager;
import com.visionassist.appspace.models.captioner.BLIPModel;
import com.visionassist.appspace.models.classifier.YOLOClassifier;
import com.visionassist.appspace.models.detector.YOLODetectorPool;
import com.visionassist.appspace.models.sttengine.SpeechRecognizer;
import com.visionassist.appspace.models.translator.CaptionTranslator;
import com.visionassist.appspace.utils.AppConfig;
import com.visionassist.appspace.utils.BackgroundTaskExecutor;
import com.visionassist.appspace.utils.Constants;
import java.util.ArrayList;
import java.util.List;

public class ModelManager {
    private static final String TAG = "ModelManager";

    // Model instances
    private YOLODetectorPool detector;
    private BLIPModel captioner;
    private YOLOClassifier classifier;
    private SpeechRecognizer speechRecognizer;
    private CaptionTranslator translator;

    // Notifications for errors on loading models
    private InfoNotificationManager infoNotificationManager;
    private List<Runnable> pendingNotifications;

    // Callback for completion
    private Runnable onAllModelsLoadedCallback;

    // Error tracking
    private int loadedModelsCount = 0;

    // Translator download state
    private Runnable translatorDownloadNotification = null;

    // Background task executor
    private BackgroundTaskExecutor executor;
    private Handler mainHandler;

    public ModelManager() {
        this.executor = BackgroundTaskExecutor.getInstance();
        this.mainHandler = new Handler(Looper.getMainLooper());
        this.pendingNotifications = new ArrayList<>();
    }

    public void loadAssets(Runnable onComplete) {
        this.infoNotificationManager = new InfoNotificationManager(PhoneStatusMonitor.getInstance().getCurrentActivity());
        Log.d(TAG, "Starting to load all AI models...");

        this.onAllModelsLoadedCallback = onComplete;
        this.loadedModelsCount = 0;
        this.pendingNotifications.clear();

        // Start loading models in sequence
        //loadDetectorAcc();
        //loadDetectorSpeed();
        loadDetector();
        loadCaptioner();
        loadClassifier();
        loadSpeechRecognizer();
        loadTranslator();
        waitForAll();
    }

    private void loadDetector() {
        executor.executeAsync(
                () -> {
                    detector = new YOLODetectorPool(PhoneStatusMonitor.getInstance().getCurrentActivity());
                    // Determine motion state (use gyroscope/accelerometer data if available)
                    boolean success = detector.initialize();
                    return success ? 0 : -1;
                },
                new BackgroundTaskExecutor.TaskCallback<>() {
                    @Override
                    public void onSuccess(Integer result) {
                        if (result == -1) {
                            Log.e(TAG, "Detector pool failed to initialize");
                            handleCriticalError(Constants.DETECTOR_LOAD_ERROR);
                        } else {
                            Log.d(TAG, "Detector pool initialized successfully");
                            loadedModelsCount++;
                        }
                    }

                    @Override
                    public void onError(Exception e) {
                        Log.e(TAG, "Error loading detector pool", e);
                        handleCriticalError(Constants.DETECTOR_LOAD_ERROR);
                    }
                }
        );
    }

    private void loadCaptioner() {
        Log.d(TAG, "Loading BLIP Captioner...");

        executor.executeAsync(
                () -> {
                    captioner = new BLIPModel(PhoneStatusMonitor.getInstance().getCurrentActivity());
                    return captioner.initModel();
                },
                new BackgroundTaskExecutor.TaskCallback<>() {
                    @Override
                    public void onSuccess(Integer result) {
                        if (result == -1) {
                            Log.e(TAG, "Captioner failed to load");
                            handleCriticalError(Constants.CAPTIONER_LOAD_ERROR);
                        } else {
                            Log.d(TAG, "Captioner loaded successfully");
                            loadedModelsCount++;
                        }
                    }

                    @Override
                    public void onError(Exception e) {
                        Log.e(TAG, "Captioner loading exception", e);
                        handleCriticalError(Constants.CAPTIONER_LOAD_ERROR);
                    }
                }
        );
    }

    private void loadClassifier() {
        if (!AppConfig.env_reports) loadedModelsCount++;
        else
            executor.executeAsync(
                    () -> {
                        classifier = new YOLOClassifier(PhoneStatusMonitor.getInstance().getCurrentActivity());
                        return classifier.loadModel();
                    },
                    new BackgroundTaskExecutor.TaskCallback<>() {
                        @Override
                        public void onSuccess(Integer result) {
                            if (result == -1) {
                                Log.e(TAG, "Classifier failed to load");
                                classifier = null;
                                createErrorNotification(load_sceneClassifierError(PhoneStatusMonitor.getInstance().getCurrentContext()));
                            } else {
                                Log.d(TAG, "Classifier loaded successfully");
                            }
                            loadedModelsCount++;
                        }

                        @Override
                        public void onError(Exception e) {
                            Log.e(TAG, "Classifier loading exception", e);
                            classifier = null;
                            createErrorNotification(load_sceneClassifierError(PhoneStatusMonitor.getInstance().getCurrentContext()));
                            loadedModelsCount++;
                        }
                    }
            );
    }

    private void loadSpeechRecognizer() {
        if (AppConfig.mainLanguage.getCode().equals("en")) {
            final boolean[] loaded = {false};
            executor.executeAsync(
                    () -> {
                        speechRecognizer = new SpeechRecognizer(PhoneStatusMonitor.getInstance().getCurrentActivity());
                        int loadResult = speechRecognizer.loadModel();

                        if (loadResult == -1) {
                            return -1;
                        }

                        // Wait for Vosk model to be ready (async unpacking)
                        Log.d(TAG, "Waiting for Vosk model to be ready...");

                        Handler handler = new Handler(Looper.getMainLooper());
                        Runnable checkVoskLoaded = new Runnable() {
                            @Override
                            public void run() {
                                if (speechRecognizer.isReady) {
                                    loaded[0] = speechRecognizer.getModel() == null;
                                } else {
                                    Log.e(TAG, "Sync not ready. Retrying...");
                                    handler.postDelayed(this, Constants.LOAD_CHECK_DELAY_MS);
                                }
                            }
                        };
                        handler.post(checkVoskLoaded);
                        return 0;
                    },
                    new BackgroundTaskExecutor.TaskCallback<>() {
                        @Override
                        public void onSuccess(Integer result) {
                            if (result == -1 || loaded[0]) {
                                Log.e(TAG, "Speech Recognizer failed to load");
                                speechRecognizer = null;
                                createErrorNotification(load_speechRecognizerError(PhoneStatusMonitor.getInstance().getCurrentContext()));
                            } else {
                                Log.d(TAG, "Speech Recognizer loaded successfully");
                            }
                            loadedModelsCount++;
                        }

                        @Override
                        public void onError(Exception e) {
                            Log.e(TAG, "Speech Recognizer loading exception", e);
                            speechRecognizer = null;
                            createErrorNotification(load_speechRecognizerError(PhoneStatusMonitor.getInstance().getCurrentContext()));
                            loadedModelsCount++;
                        }
                    }
            );
        } else
            loadedModelsCount++;
    }

    private void loadTranslator() {
        if (AppConfig.mainLanguage.getCode().equals("ro"))
            executor.executeAsync(
                    () -> {
                        translator = new CaptionTranslator(PhoneStatusMonitor.getInstance().getCurrentActivity());
                        return translator.initializeTranslator();
                    },
                    new BackgroundTaskExecutor.TaskCallback<>() {
                        @Override
                        public void onSuccess(Integer result) {
                            if (result == 0) {
                                // Model already downloaded
                                Log.d(TAG, "Translator loaded successfully");
                                loadedModelsCount++;
                            } else if (result == -2) {
                                // Model needs to be downloaded
                                Log.d(TAG, "Translator model needs to be downloaded");
                                createTranslatorDownloadNotification();
                            } else {
                                // Error
                                Log.e(TAG, "Translator failed to initialize");
                                translator = null;
                                createErrorNotification(load_translaterError(PhoneStatusMonitor.getInstance().getCurrentContext()));
                                loadedModelsCount++;
                            }
                        }

                        @Override
                        public void onError(Exception e) {
                            Log.e(TAG, "❌ Translator loading exception", e);
                            translator = null;
                            createErrorNotification(load_translaterError(PhoneStatusMonitor.getInstance().getCurrentContext()));
                            loadedModelsCount++;
                        }
                    }
            );
        else
            loadedModelsCount++;
    }

    private void createTranslatorDownloadNotification() {
        final Runnable[] errorNotificationWrapper = new Runnable[1];

        Runnable downloadNotification = () -> infoNotificationManager.showNotificationTwoButtons(
                "Model de traducere~Modelul de traducere în limba română nu a fost descărcat.Îl descărcați acum? (Necesită conexiune la internet)",
                "Folosiți fără",
                "Descarcă",
                // Use Without button
                () -> {
                    Log.d(TAG, "User chose to skip translator download");
                    infoNotificationManager.hideNotification();
                    pendingNotifications.remove(errorNotificationWrapper[0]);
                    processNextNotification();
                },
                // Download button
                () -> {
                    // Check internet connection
                    if (NetworkUtils.isNetworkConnected(PhoneStatusMonitor.getInstance().getCurrentActivity())) {
                        // Has internet, proceed with download
                        pendingNotifications.remove(errorNotificationWrapper[0]);
                        infoNotificationManager.hideNotification();
                        downloadTranslatorModel();
                    }
                }
        );

        errorNotificationWrapper[0] = downloadNotification;
        pendingNotifications.add(downloadNotification);
        processNextNotification();
    }

    private void downloadTranslatorModel() {
        Log.d(TAG, "Downloading translator model...");

        executor.executeAsync(
                () -> {
                    int downloadResult = translator.downloadModel();

                    if (downloadResult == -1) {
                        return -1;
                    }

                    // Model downloaded, try to initialize again
                    return translator.initializeTranslator();
                },
                new BackgroundTaskExecutor.TaskCallback<>() {
                    @Override
                    public void onSuccess(Integer result) {

                        mainHandler.postDelayed(() -> {
                            if (result == -1) {
                                // Download failed
                                Log.e(TAG, "Translator download failed");
                                translatorDownloadNotification = () -> infoNotificationManager.showNotification(
                                        load_translaterError(PhoneStatusMonitor.getInstance().getCurrentContext()),
                                        () -> {
                                            infoNotificationManager.hideNotification();
                                            pendingNotifications.remove(translatorDownloadNotification);
                                            processNextNotification();
                                        },
                                        "OK"
                                );
                                translatorDownloadNotification.run();
                            } else {
                                // Download successful
                                Log.d(TAG, "Translator downloaded and loaded successfully");
                                loadedModelsCount++;
                                processNextNotification();
                            }
                        }, 300);
                    }

                    @Override
                    public void onError(Exception e) {

                        mainHandler.postDelayed(() -> {
                            Log.e(TAG, "Translator download exception", e);
                            translatorDownloadNotification = () -> infoNotificationManager.showNotification(
                                    load_translaterError(PhoneStatusMonitor.getInstance().getCurrentContext()),
                                    () -> {
                                        infoNotificationManager.hideNotification();
                                        pendingNotifications.remove(translatorDownloadNotification);
                                        processNextNotification();
                                    },
                                    "OK"
                            );
                            translatorDownloadNotification.run();
                        }, 300);
                    }
                }
        );
    }

    private void createErrorNotification(String message) {
        final Runnable[] errorNotificationWrapper = new Runnable[1];

        Runnable errorNotification = () -> infoNotificationManager.showNotification(
                message,
                () -> {
                    infoNotificationManager.hideNotification();
                    pendingNotifications.remove(errorNotificationWrapper[0]);
                    processNextNotification();
                },
                "OK"
        );

        errorNotificationWrapper[0] = errorNotification;
        pendingNotifications.add(errorNotification);
    }

    private void waitForAll() {
        Runnable checkLoadedAssets = new Runnable() {
            @Override
            public void run() {
                if (loadedModelsCount == Constants.MODELS_COUNT) {
                    processNextNotification();
                } else {
                    Log.e(TAG, "Models not loaded yet. Retrying...");
                    mainHandler.postDelayed(this, Constants.LOAD_CHECK_DELAY_MS);
                }
            }
        };
        mainHandler.post(checkLoadedAssets);
    }

    private void processNextNotification() {
        if (!pendingNotifications.isEmpty()) {
            Runnable nextNotification = pendingNotifications.get(0);
            mainHandler.post(nextNotification);
        } else {
            mainHandler.removeCallbacksAndMessages(null);
            mainHandler.post(onAllModelsLoadedCallback);
        }
    }

    private void handleCriticalError(int errorCode) {
        Log.e(TAG, "Critical error occurred, code: " + errorCode);

        PhoneStatusMonitor phoneMonitor = PhoneStatusMonitor.getInstance();
        ErrorDialogManager errorDialog = new ErrorDialogManager(phoneMonitor.getCurrentActivity());
        errorDialog.setupDialog(errorCode);
        phoneMonitor.shutdownApp(errorDialog, phoneMonitor.getCurrentActivity());
    }

    public YOLODetectorPool getDetector() {
        return detector;
    }

    public BLIPModel getCaptioner() {
        return captioner;
    }

    public YOLOClassifier getClassifier() {
        return classifier;
    }

    public SpeechRecognizer getSpeechRecognizer() {
        return speechRecognizer;
    }

    public CaptionTranslator getTranslator() {
        return translator;
    }

    /**
     * Cleanup all loaded models and free resources
     */
    public void cleanup() {
        Log.d(TAG, "Cleaning up all models...");

        if (detector != null) {
            try {
                detector.cleanup();
                Log.d(TAG, "Detector closed");
            } catch (Exception e) {
                Log.e(TAG, "Error closing detector", e);
            }
            detector = null;
        }

        if (captioner != null) {
            try {
                captioner.close();
                Log.d(TAG, "Captioner closed");
            } catch (Exception e) {
                Log.e(TAG, "Error closing captioner", e);
            }
            captioner = null;
        }

        if (classifier != null) {
            try {
                classifier.close();
                Log.d(TAG, "Classifier closed");
            } catch (Exception e) {
                Log.e(TAG, "Error closing classifier", e);
            }
            classifier = null;
        }

        if (speechRecognizer != null) {
            try {
                speechRecognizer.close();
                Log.d(TAG, "Speech recognizer closed");
            } catch (Exception e) {
                Log.e(TAG, "Error closing speech recognizer", e);
            }
            speechRecognizer = null;
        }

        if (translator != null) {
            try {
                translator.close();
                Log.d(TAG, "Translator closed");
            } catch (Exception e) {
                Log.e(TAG, "Error closing translator", e);
            }
            translator = null;
        }

        Log.d(TAG, "All models cleaned up successfully");
    }
}