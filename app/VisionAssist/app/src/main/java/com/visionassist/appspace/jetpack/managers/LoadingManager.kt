package com.visionassist.appspace.jetpack.managers

import android.content.Context
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.platform.ComposeView
import androidx.compose.ui.res.stringResource
import com.visionassist.appspace.PhoneStatusMonitor
import com.visionassist.appspace.R
import com.visionassist.appspace.jetpack.design.LoadingComponent
import com.visionassist.appspace.utils.load_loadingText

class LoadingManager(
    private val loadingBox: ComposeView,
    private val initMessage: Boolean = false,
    private val context: Context
) {
    private var isVisibleState = mutableStateOf(false)
    private var currentMessageState = mutableStateOf("Please wait")

    /**
     * Initializes the ComposeView with the LoadingComponent.
     * This needs to be called once in onCreate.
     */
    fun setupLoadingBox() {
        loadingBox.setContent {
            if (initMessage) {
                LoadingComponent(
                    isVisible = isVisibleState.value,
                    loadingText = currentMessageState.value
                )
            } else {
                if(!PhoneStatusMonitor.getInstance().profileLoaded) {
                    LoadingComponent(
                        isVisible = isVisibleState.value,
                        loadingText = stringResource(R.string.wait_en)
                    )
                }
                else
                {
                    LoadingComponent(
                        isVisible = isVisibleState.value,
                        loadingText = load_loadingText(context)
                    )
                }
            }
        }
    }

    /**
     * Displays the loading screen with the default message (mainLanguage)
     */
    fun showLoading() {
        if (!initMessage) {
            isVisibleState.value = true
        }
    }

    /**
     * Displays the loading screen with a custom message.
     */
    fun showLoading(message: String) {
        if (initMessage) {
            currentMessageState.value = message
            isVisibleState.value = true
        }
    }

    fun changeText(message: String) {
        if (initMessage) {
            currentMessageState.value = message
        }
    }

    /**
     * Hides the loading screen with fade out animation.
     */
    fun hideLoading() {
        isVisibleState.value = false
    }
}