package com.visionassist.appspace.jetpack.managers

import android.app.Activity
import android.view.ViewGroup
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.platform.ComposeView
import androidx.compose.ui.platform.ViewCompositionStrategy
import com.visionassist.appspace.PhoneStatusMonitor
import com.visionassist.appspace.jetpack.design.ErrorDialog
import com.visionassist.appspace.utils.load_errorText

class ErrorDialogManager(
    private val activity: Activity
) {
    private var isVisibleState = mutableStateOf(false)
    private var composeView: ComposeView? = null
    private var isSetup = false

    /**
     * Setup dialog with custom message and error code
     * Call this once during initialization
     */
    fun setupDialog(errorCode: Int) {
        val message : String = if (!PhoneStatusMonitor.getInstance().profileLoaded)
            "A critical error has occurred and the application needs to close to protect the integrity of your data. If this happens again, contact support with the code below"
        else load_errorText(PhoneStatusMonitor.getInstance().currentContext)
        attachDialogToActivity(message,errorCode)
        isSetup = true
    }

    /**
     * Show the dialog (must call setupDialog first)
     */
    fun showDialog() {
        if (!isSetup) {
            throw IllegalStateException("Must call setupDialog() before showDialog()")
        }
        isVisibleState.value = true
    }

    /**
     * Attach the ComposeView to the activity's root view
     */
    private fun attachDialogToActivity(message: String, errorCode: Int) {
        if (composeView == null) {
            composeView = ComposeView(activity).apply {
                setViewCompositionStrategy(ViewCompositionStrategy.DisposeOnDetachedFromWindow)

                setContent {
                    ErrorDialog(
                        context = activity,
                        isVisible = isVisibleState.value,
                        message = message,
                        errorCode = errorCode
                    )
                }

                // Add to activity's root view
                layoutParams = ViewGroup.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT
                )
            }

            // Get the activity's root view and add our ComposeView on top
            val rootView = activity.window.decorView.findViewById<ViewGroup>(android.R.id.content)
            rootView.addView(composeView)
        }
    }
}