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
        val quickAction = intent?.getStringExtra("QUICK_ACTION") ?: "Disabled"

        Log.d(TAG, "LockScreenService started with action: $quickAction")

        if (quickAction == "Disabled") {
            Log.d(TAG, "Quick action is disabled, stopping service")
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

        // Create intent to launch MainActivity with quick action target
        val launchIntent = Intent(this, MainActivity::class.java).apply {
            putExtra("QUICK_ACTION_TARGET", quickAction)
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

        // Build notification
        val notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("VisionAssist Quick Action")
            .setContentText("Tap to open $quickAction")
            .setSmallIcon(R.drawable.vision_assist_logo)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .setVisibility(NotificationCompat.VISIBILITY_PUBLIC)
            .setAutoCancel(false)
            .build()

        startForeground(NOTIFICATION_ID, notification)

        Log.d(TAG, "Foreground notification started for: $quickAction")

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

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        super.onDestroy()
        Log.d(TAG, "LockScreenService destroyed")
    }

    companion object {
        fun startService(context: android.content.Context, quickAction: String) {
            val intent = Intent(context, LockScreenService::class.java).apply {
                putExtra("QUICK_ACTION", quickAction)
            }

            if (Constants.API_LEVEL >= Build.VERSION_CODES.O) {
                context.startForegroundService(intent)
            } else {
                context.startService(intent)
            }
        }

        fun stopService(context: android.content.Context) {
            val intent = Intent(context, LockScreenService::class.java)
            context.stopService(intent)
        }
    }
}