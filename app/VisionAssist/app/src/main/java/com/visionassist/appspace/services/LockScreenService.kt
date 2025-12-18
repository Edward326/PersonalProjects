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
import androidx.core.content.edit

class LockScreenService : Service() {

    private val TAG = "LockScreenService"
    private val CHANNEL_ID = "quick_action_channel"
    private val NOTIFICATION_ID = 1001

    companion object {
        private const val PREFS_NAME = "VisionAssistPrefs"
        private const val KEY_QUICK_ACTION_INDEX = "quick_action_index"

        const val ACTION_DISABLED = 0
        const val ACTION_DISABLE_SERVICE = "com.visionassist.DISABLE_QUICK_ACTION"  // Action for disable button

        fun startService(context: android.content.Context, actionIndex: Int) {
            saveQuickActionIndex(context, actionIndex)

            if (actionIndex == ACTION_DISABLED) {
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

        fun stopService(context: android.content.Context) {
            saveQuickActionIndex(context, ACTION_DISABLED)
            val intent = Intent(context, LockScreenService::class.java)
            context.stopService(intent)
        }

        fun getCurrentQuickActionIndex(context: android.content.Context): Int {
            val prefs = context.getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
            return prefs.getInt(KEY_QUICK_ACTION_INDEX, ACTION_DISABLED)
        }

        private fun saveQuickActionIndex(context: android.content.Context, actionIndex: Int) {
            val prefs = context.getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
            prefs.edit { putInt(KEY_QUICK_ACTION_INDEX, actionIndex) }
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // Check if this is a disable action from notification button
        if (intent?.action == ACTION_DISABLE_SERVICE) {
            Log.d(TAG, "Disable action received from notification")
            stopService(this)
            return START_NOT_STICKY
        }

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

        // Main action: Launch MainActivity with quick action index
        val launchIntent = Intent(this, MainActivity::class.java).apply {
            putExtra("QUICK_ACTION_INDEX", quickActionIndex)
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP)
        }

        val launchPendingIntent = PendingIntent.getActivity(
            this,
            0,
            launchIntent,
            if (Constants.API_LEVEL >= Build.VERSION_CODES.M) {
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
            } else {
                PendingIntent.FLAG_UPDATE_CURRENT
            }
        )

        // Disable button action: Stop service
        val disableIntent = Intent(this, LockScreenService::class.java).apply {
            action = ACTION_DISABLE_SERVICE
        }

        val disablePendingIntent = PendingIntent.getService(
            this,
            1,
            disableIntent,
            if (Constants.API_LEVEL >= Build.VERSION_CODES.M) {
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
            } else {
                PendingIntent.FLAG_UPDATE_CURRENT
            }
        )

        val actionName = getActionName(quickActionIndex)

        // Build notification with disable button
        val notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("VisionAssist Quick Action")
            .setContentText("Tap to open $actionName")
            .setSmallIcon(R.drawable.vision_assist_logo_resized)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setContentIntent(launchPendingIntent)
            .setOngoing(true)
            .setVisibility(NotificationCompat.VISIBILITY_PUBLIC)
            .setAutoCancel(false)
            // Add disable button
            .addAction(
                R.drawable.ic_close,  // You need to add this icon or use android.R.drawable.ic_menu_close_clear_cancel
                "Disable",
                disablePendingIntent
            )
            .setShowWhen(false)
            .build()

        startForeground(NOTIFICATION_ID, notification)

        Log.d(TAG, "Foreground notification started for index: $quickActionIndex")

        return START_STICKY
    }

    private fun createNotificationChannel() {
        if (Constants.API_LEVEL >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "VisionAssist Quick Action",
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
}