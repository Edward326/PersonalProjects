package com.visionassist.appspace.services

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log

/**
 * OPTIONAL: Boot Completed Receiver
 *
 * This receiver restarts the LockScreenService after device reboot
 * if Quick Action was enabled before reboot.
 *
 * To use this:
 * 1. Uncomment the receiver declaration in AndroidManifest.xml
 * 2. Add RECEIVE_BOOT_COMPLETED permission to AndroidManifest.xml
 * 3. Store the selected Quick Action in SharedPreferences when changed
 * 4. Read it here and restart the service if needed
 */
class BootCompletedReceiver : BroadcastReceiver() {

    private val TAG = "BootCompletedReceiver"

    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action == Intent.ACTION_BOOT_COMPLETED ||
            intent.action == "android.intent.action.QUICKBOOT_POWERON") {

            Log.d(TAG, "Device boot completed, checking Quick Action status")

            // Read Quick Action setting from SharedPreferences
            val prefs = context.getSharedPreferences("VisionAssistPrefs", Context.MODE_PRIVATE)
            val quickAction = prefs.getString("quick_action", "Disabled") ?: "Disabled"

            Log.d(TAG, "Quick Action setting: $quickAction")

            if (quickAction != "Disabled") {
                Log.d(TAG, "Restarting LockScreenService with action: $quickAction")
                LockScreenService.startService(context, quickAction)
            } else {
                Log.d(TAG, "Quick Action is disabled, not starting service")
            }
        }
    }
}