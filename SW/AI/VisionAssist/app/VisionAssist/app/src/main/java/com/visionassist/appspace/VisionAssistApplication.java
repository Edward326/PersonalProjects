package com.visionassist.appspace;

import android.app.Application;
import android.util.Log;

public class VisionAssistApplication extends Application {
    private static final String TAG = "VisionAssistApp";

    @Override
    public void onCreate() {
        super.onCreate();
        Log.d(TAG, "Application starting");

        // Initialize the phone status monitor
        PhoneStatusMonitor.initialize(this);

        // Add other app-wide initializations here
    }

    @Override
    public void onTerminate() {
        super.onTerminate();

        // Cleanup
        PhoneStatusMonitor instance = PhoneStatusMonitor.getInstance();
        if (instance != null) {
            instance.exitApp();
        }

        Log.d(TAG, "Application terminating");
    }
}