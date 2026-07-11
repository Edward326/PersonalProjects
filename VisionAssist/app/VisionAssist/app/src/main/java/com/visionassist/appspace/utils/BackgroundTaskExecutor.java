package com.visionassist.appspace.utils;

import android.os.Handler;
import android.os.Looper;
import android.util.Log;

public class BackgroundTaskExecutor {
    private static final String TAG = "BackgroundTaskExecutor";

    private static BackgroundTaskExecutor instance;
    private final Handler mainHandler = new Handler(Looper.getMainLooper());

    public interface TaskCallback<T> {
        void onSuccess(T result)throws Exception;

        void onError(Exception e);
    }


    public interface BackgroundTask<T> {
        T execute() throws Exception;
    }


    private BackgroundTaskExecutor() {
    }


    public static synchronized BackgroundTaskExecutor getInstance() {
        if (instance == null) {
            instance = new BackgroundTaskExecutor();
        }
        return instance;
    }

    public <T> void executeAsync(BackgroundTask<T> task, TaskCallback<T> callback) {
        new Thread(() -> {
            try {
                T result = task.execute();
                mainHandler.post(() -> {
                    try {
                        callback.onSuccess(result);
                    } catch (Exception e) {
                        mainHandler.post(() -> callback.onError(e));
                    }
                });
            } catch (Exception e) {
                mainHandler.post(() -> callback.onError(e));
            }
        }).start();
    }

    public void executeAsync(Runnable task, Runnable onComplete, Runnable onError) {
        new Thread(() -> {
            try {
                task.run();
                mainHandler.post(onComplete);
            } catch (Exception e) {
                Log.e(TAG, "Background task failed", e);
                mainHandler.post(onError);
            }
        }).start();
    }
}