package com.visionassist.appspace.services

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log

/**
 * Boot Completed Receiver
 *
 * This receiver restarts the LockScreenService after device reboot
 * if Quick Action was enabled before reboot.
 */
class BootCompletedReceiver : BroadcastReceiver() {

    private val TAG = "BootCompletedReceiver"

    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action == Intent.ACTION_BOOT_COMPLETED ||
            intent.action == "android.intent.action.QUICKBOOT_POWERON") {

            Log.d(TAG, "Device boot completed, checking Quick Action status")

            // Read Quick Action index from SharedPreferences
            val quickActionIndex = LockScreenService.getCurrentQuickActionIndex(context)

            Log.d(TAG, "Quick Action index: $quickActionIndex")

            if (quickActionIndex != LockScreenService.ACTION_DISABLED) {
                Log.d(TAG, "Restarting LockScreenService with action index: $quickActionIndex")
                LockScreenService.startService(context, quickActionIndex)
            } else {
                Log.d(TAG, "Quick Action is disabled (index 0), not starting service")
            }
        }
    }
}