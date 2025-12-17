package com.visionassist.appspace.services

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import com.visionassist.appspace.R
import com.visionassist.appspace.activities.main.MainActivity
import com.visionassist.appspace.utils.Constants

class LockScreenService : Service() {

    private val TAG = "LockScreenService"
    private val CHANNEL_ID = "quick_action_channel"
    private val NOTIFICATION_ID = 1001

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val quickActionIndex = intent?.getIntExtra("QUICK_ACTION_INDEX", 0) ?: 0

        Log.d(TAG, "LockScreenService started with action index: $quickActionIndex")

        if (quickActionIndex == 0) {
            Log.d(TAG, "Quick action is disabled (index 0), stopping service")
            if (Constants.API_LEVEL >= Build.VERSION_CODES.N) {
                stopForeground(STOP_FOREGROUND_REMOVE)
            } else {
                @Suppress("DEPRECATION")
                stopForeground(true)
            }
            stopSelf()
            return START_NOT_STICKY
        }

        createNotificationChannel()

        // Create intent to launch MainActivity with quick action index
        val launchIntent = Intent(this, MainActivity::class.java).apply {
            putExtra("QUICK_ACTION_INDEX", quickActionIndex)
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP)
        }

        val pendingIntent = PendingIntent.getActivity(
            this,
            0,
            launchIntent,
            if (Constants.API_LEVEL >= Build.VERSION_CODES.M) {
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
            } else {
                PendingIntent.FLAG_UPDATE_CURRENT
            }
        )

        // Get action name for display
        val actionName = getActionName(quickActionIndex)

        // Build notification
        val notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("VisionAssist Quick Action")
            .setContentText("Tap to open $actionName")
            .setSmallIcon(R.drawable.vision_assist_logo)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .setVisibility(NotificationCompat.VISIBILITY_PUBLIC)
            .setAutoCancel(false)
            .build()

        startForeground(NOTIFICATION_ID, notification)

        Log.d(TAG, "Foreground notification started for index: $quickActionIndex")

        return START_STICKY
    }

    private fun createNotificationChannel() {
        if (Constants.API_LEVEL >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Quick Action",
                NotificationManager.IMPORTANCE_HIGH
            ).apply {
                description = "Lock screen quick action access"
                lockscreenVisibility = Notification.VISIBILITY_PUBLIC
            }

            val manager = getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(channel)

            Log.d(TAG, "Notification channel created")
        }
    }

    private fun getActionName(index: Int): String {
        return when (index) {
            1 -> "Detection-static"
            2 -> "Detection-dynamic"
            3 -> "Caption"
            else -> "Quick Action"
        }
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        super.onDestroy()
        Log.d(TAG, "LockScreenService destroyed")
    }

    companion object {
        private const val PREFS_NAME = "VisionAssistPrefs"
        private const val KEY_QUICK_ACTION_INDEX = "quick_action_index"

        // Quick Action Indexes
        const val ACTION_DISABLED = 0

        /**
         * Start the service with a quick action index
         * @param context Application context
         * @param actionIndex 0=Disabled, 1=Detection-static, 2=Detection-dynamic, 3=Caption
         */
        fun startService(context: android.content.Context, actionIndex: Int) {
            // Save to SharedPreferences for persistence
            saveQuickActionIndex(context, actionIndex)

            if (actionIndex == ACTION_DISABLED) {
                // If disabled, just stop the service
                stopService(context)
                return
            }

            val intent = Intent(context, LockScreenService::class.java).apply {
                putExtra("QUICK_ACTION_INDEX", actionIndex)
            }

            if (Constants.API_LEVEL >= Build.VERSION_CODES.O) {
                context.startForegroundService(intent)
            } else {
                context.startService(intent)
            }
        }

        /**
         * Stop the service
         */
        fun stopService(context: android.content.Context) {
            // Clear from SharedPreferences
            saveQuickActionIndex(context, ACTION_DISABLED)

            val intent = Intent(context, LockScreenService::class.java)
            context.stopService(intent)
        }

        /**
         * Get the current quick action index
         * @return 0=Disabled, 1=Detection-static, 2=Detection-dynamic, 3=Caption
         */
        fun getCurrentQuickActionIndex(context: android.content.Context): Int {
            val prefs = context.getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
            return prefs.getInt(KEY_QUICK_ACTION_INDEX, ACTION_DISABLED)
        }

        /**
         * Save the quick action index to SharedPreferences
         */
        private fun saveQuickActionIndex(context: android.content.Context, actionIndex: Int) {
            val prefs = context.getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
            prefs.edit().putInt(KEY_QUICK_ACTION_INDEX, actionIndex).apply()
        }
    }
}