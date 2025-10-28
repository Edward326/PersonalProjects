package com.visionassist.appspace.jetpack.managers

import android.content.Context
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.platform.ComposeView
import androidx.compose.ui.res.stringResource
import com.visionassist.appspace.PhoneStatusMonitor
import com.visionassist.appspace.R
import com.visionassist.appspace.jetpack.design.PermissionDialog
import com.visionassist.appspace.utils.load_alwaysAllowPermDialogBox
import com.visionassist.appspace.utils.load_settingsButton
import com.visionassist.appspace.utils.load_settingsInfo

class PermissionDialogManager(
    private val dialogBox: ComposeView,
    private val showSettings: Boolean = false,
    private val context: Context
) {
    private var isVisibleState = mutableStateOf(false)
    private var onOkClickHandler: (() -> Unit)? = null

    fun setupDialog(onOkClick: () -> Unit) {
        this.onOkClickHandler = onOkClick

        if (!PhoneStatusMonitor.getInstance().profileLoaded) {
            if (!showSettings) {
                dialogBox.setContent {
                    PermissionDialog(
                        context = context,
                        isVisible = isVisibleState.value,
                        message = stringResource(R.string.always_allow_en),
                        buttonText = stringResource(R.string.ok),
                        onOkClick = {
                            onOkClickHandler?.invoke()
                        }
                    )
                }
            } else {
                dialogBox.setContent {
                    PermissionDialog(
                        context = context,
                        isVisible = isVisibleState.value,
                        message = stringResource(R.string.settings_open_en),
                        buttonText = stringResource(R.string.settings_button_en),
                        onOkClick = {
                            onOkClickHandler?.invoke()
                        }
                    )
                }
            }
        } else {
            if (!showSettings) {
                dialogBox.setContent {
                    PermissionDialog(
                        context = context,
                        isVisible = isVisibleState.value,
                        message = load_alwaysAllowPermDialogBox(context),
                        buttonText = stringResource(R.string.ok),
                        onOkClick = {
                            onOkClickHandler?.invoke()
                        }
                    )
                }
            } else {
                dialogBox.setContent {
                    PermissionDialog(
                        context = context,
                        isVisible = isVisibleState.value,
                        message = load_settingsInfo(context),
                        buttonText = load_settingsButton(context),
                        onOkClick = {
                            onOkClickHandler?.invoke()
                        }
                    )
                }
            }
        }
    }

    fun showDialog() {
        isVisibleState.value = true
    }

    fun hideDialog() {
        isVisibleState.value = false
    }
}