package com.visionassist.appspace.utils

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.util.Log
import androidx.compose.runtime.MutableState
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.TextUnit
import androidx.core.content.ContextCompat.getSystemService
import com.visionassist.appspace.PhoneStatusMonitor
import com.visionassist.appspace.R
import com.visionassist.appspace.activities.tabs.home.caption.SemanticHash.similarity
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

data class Language(
    var code: String,
    var name: String,
    var country: String
)

enum class DetectionOption {
    LIVE, STATIC
}

data class CustomColorSchema(
    val color: Color,
    val fontSize: TextUnit,
    val fontFamily: FontFamily
)

data class TypewriterColorSchema(
    val wordColorSchema: CustomColorSchema,
    val outlinedWordColorSchema: CustomColorSchema
)

val robotoRegular = FontFamily(
    Font(R.font.roboto_regular)
)

val robotoSemibold = FontFamily(
    Font(R.font.roboto_semibold)
)

val robotoLight = FontFamily(
    Font(R.font.roboto_light)
)

val robotoBold = FontFamily(
    Font(R.font.roboto_bold)
)

val robotoExtraBold = FontFamily(
    Font(R.font.roboto_extrabold)
)

val robotoMediumItalic = FontFamily(
    Font(R.font.roboto_mediumitalic)
)

val robotoExtraBoldItalic = FontFamily(
    Font(R.font.roboto_extrabolditalic)
)

fun load_loadingText(context: Context): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.wait_en)
        "ro" -> context.getString(R.string.wait_ro)
        else -> context.getString(R.string.wait_en)
    }
}

fun load_loadingVerifying(context: Context): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.verifying_en)
        "ro" -> context.getString(R.string.verifying_ro)
        else -> context.getString(R.string.verifying_en)
    }
}

fun load_loadingUploading(context: Context): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.uploading_loading_en)
        "ro" -> context.getString(R.string.uploading_loading_ro)
        else -> context.getString(R.string.uploading_loading_en)
    }
}

fun load_creatingAccount(context: Context): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.create_account_en)
        "ro" -> context.getString(R.string.create_account_ro)
        else -> context.getString(R.string.create_account_en)
    }
}

fun load_createdAccount(context: Context): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.created_account_en)
        "ro" -> context.getString(R.string.created_account_ro)
        else -> context.getString(R.string.created_account_en)
    }
}

fun load_tempErrorText(context: Context, codeLanguage: String): String {
    return when (codeLanguage) {
        "en" -> context.getString(R.string.temp_error_en)
        "ro" -> context.getString(R.string.temp_error_ro)
        else -> context.getString(R.string.temp_error_en)
    }
}

fun load_batteryLowText(context: Context, codeLanguage: String): String {
    return when (codeLanguage) {
        "en" -> context.getString(R.string.battery_low_en)
        "ro" -> context.getString(R.string.battery_low_ro)
        else -> context.getString(R.string.battery_low_en)
    }
}

fun load_criticalWarning(context: Context, codeLanguage: String): String {
    return when (codeLanguage) {
        "en" -> context.getString(R.string.critical_warning_en)
        "ro" -> context.getString(R.string.critical_warning_ro)
        else -> context.getString(R.string.critical_warning_en)
    }
}

fun load_alwaysAllowPermDialogBox(context: Context): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.always_allow_en)
        "ro" -> context.getString(R.string.always_allow_ro)
        else -> context.getString(R.string.always_allow_en)
    }
}

fun load_settingsInfo(context: Context): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.settings_open_en)
        "ro" -> context.getString(R.string.settings_open_ro)
        else -> context.getString(R.string.settings_open_en)
    }
}

fun load_settingsButton(context: Context): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.settings_button_en)
        "ro" -> context.getString(R.string.settings_button_ro)
        else -> context.getString(R.string.settings_button_en)
    }
}

fun load_permissionActivityWarning(context: Context): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.audio_permission_warning_en)
        "ro" -> context.getString(R.string.audio_permission_warning_ro)
        else -> context.getString(R.string.audio_permission_warning_en)
    }
}

fun load_permissionInfo(context: Context, case: String): String {
    when (case) {
        "camera" ->
            return when (AppConfig.mainLanguage.code) {
                "en" -> context.getString(R.string.camera_permission_info_en)
                "ro" -> context.getString(R.string.camera_permission_info_ro)
                else -> context.getString(R.string.camera_permission_info_en)
            }

        "storage" ->
            return when (AppConfig.mainLanguage.code) {
                "en" -> context.getString(R.string.storage_permission_info_en)
                "ro" -> context.getString(R.string.storage_permission_info_ro)
                else -> context.getString(R.string.storage_permission_info_en)
            }

        "microphone" ->
            return when (AppConfig.mainLanguage.code) {
                "en" -> context.getString(R.string.microphone_permission_info_en)
                "ro" -> context.getString(R.string.microphone_permission_info_ro)
                else -> context.getString(R.string.microphone_permission_info_en)
            }
    }
    return "none"
}

fun load_loadProfileText(context: Context): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.load_profile_info_en)
        "ro" -> context.getString(R.string.load_profile_info_ro)
        else -> context.getString(R.string.load_profile_info_en)
    }
}

fun load_newProfileText(context: Context): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.new_profile_info_en)
        "ro" -> context.getString(R.string.new_profile_info_ro)
        else -> context.getString(R.string.new_profile_info_en)
    }
}

fun load_errorText(context: Context): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.exit_error_en)
        "ro" -> context.getString(R.string.exit_error_ro)
        else -> context.getString(R.string.exit_error_en)
    }
}

fun load_errorTextBlind(context: Context, exitCode: Int): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.exit_error2_en) + exitCode.toString()
        "ro" -> context.getString(R.string.exit_error2_ro) + exitCode.toString()
        else -> context.getString(R.string.exit_error2_en) + exitCode.toString()
    }
}

fun haptic_model0(): LongArray {
    return longArrayOf(0, 200)
}

fun vibrate(pattern: LongArray) {
    val vibrator = getSystemService(
        PhoneStatusMonitor.getInstance().currentContext,
        Vibrator::class.java
    ) as Vibrator
    if (Constants.API_LEVEL >= Build.VERSION_CODES.O) {
        vibrator.vibrate(
            VibrationEffect.createWaveform(pattern, -1)
        )
    }
}

fun load_profileSelectionButton(context: Context, case: Boolean): String {
    return if (case) {
        when (AppConfig.mainLanguage.code) {
            "en" -> context.getString(R.string.load_profile_en)
            "ro" -> context.getString(R.string.load_profile_ro)
            else -> context.getString(R.string.load_profile_en)
        }
    } else {
        when (AppConfig.mainLanguage.code) {
            "en" -> context.getString(R.string.new_profile_en)
            "ro" -> context.getString(R.string.new_profile_ro)
            else -> context.getString(R.string.new_profile_en)
        }
    }
}

fun load_infoLoadProfileActivity(): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> "Please select the '${Constants.PROFILE_FOLDER_NAME}' folder from your storage to load your profile"
        "ro" -> "Vă rugăm să selectați folderul '${Constants.PROFILE_FOLDER_NAME}' din fișierele dvs. pentru a încărca profilul"
        else -> "Please select the '${Constants.PROFILE_FOLDER_NAME}' folder from your storage to load your profile"
    }
}

fun load_errorLocalLoadProfileActivity(errorCode: Int): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> "Error was encountered while fetching the profile\n\n@(Error code: ${errorCode})"
        "ro" -> "A apărut o eroare în timpul încărcării profilului dvs.\n\n@(Error code: ${errorCode})"
        else -> "Error was encountered while fetching the profile\n\n@(Error code: ${errorCode})"
    }
}

fun load_successLocalLoadProfileActivity(errorCode: Int): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> "Profile imported successfully\n\n@(Exit code: ${errorCode})"
        "ro" -> "Profil încărcat cu succes\n\n@(Exit code: ${errorCode})"
        else -> "Profile imported successfully\n\n@(Exit code: ${errorCode})"
    }
}

fun load_noInternet(context: Context): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.network_miss_en)
        "ro" -> context.getString(R.string.network_miss_ro)
        else -> context.getString(R.string.network_miss_en)
    }
}

fun load_accountNE(context: Context): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.account_ne_en)
        "ro" -> context.getString(R.string.account_ne_ro)
        else -> context.getString(R.string.account_ne_en)
    }
}

fun load_genericErrorLoad(context: Context): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.generic_error_en)
        "ro" -> context.getString(R.string.generic_error_ro)
        else -> context.getString(R.string.generic_error_en)
    }
}

fun load_passChangedSuccess(emailInput: String): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> "Password change request sent to\n${emailInput}"
        "ro" -> "Cerere de schimbare a parolei trimisă către\n${emailInput}"
        else -> "Password change request sent to\n${emailInput}"
    }
}

fun load_profileImportedSuccess(): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> "Profile imported successfully"
        "ro" -> "Profilul a fost importat cu succes"
        else -> "Profile imported successfully"
    }
}

fun load_emailAlreadyExists(context: Context): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.already_exists_account_en)
        "ro" -> context.getString(R.string.already_exists_account_ro)
        else -> context.getString(R.string.already_exists_account_en)
    }
}

fun load_invalidEmail(context: Context): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.invalid_email_en)
        "ro" -> context.getString(R.string.invalid_email_ro)
        else -> context.getString(R.string.invalid_email_en)
    }
}

fun load_genericErrorNew(context: Context): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.generic_error2_en)
        "ro" -> context.getString(R.string.generic_error2_ro)
        else -> context.getString(R.string.generic_error2_en)
    }
}

fun load_invalidCombination(context: Context): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.invalid_chars_en)
        "ro" -> context.getString(R.string.invalid_chars_ro)
        else -> context.getString(R.string.invalid_chars_en)
    }
}

fun load_contributeResearch(context: Context): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.contribute_en)
        "ro" -> context.getString(R.string.contribute_ro)
        else -> context.getString(R.string.contribute_en)
    }
}

fun load_whatsYourName(context: Context): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.hello_en)
        "ro" -> context.getString(R.string.hello_ro)
        else -> context.getString(R.string.hello_en)
    }
}

fun load_howOldAreYou(context: Context): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.old_en)
        "ro" -> context.getString(R.string.old_ro)
        else -> context.getString(R.string.old_en)
    }
}

fun load_whatTypeOfVision(context: Context): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.visual_condition_en)
        "ro" -> context.getString(R.string.visual_condition_ro)
        else -> context.getString(R.string.visual_condition_en)
    }
}

fun load_agreeButton(): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> "Agree"
        "ro" -> "De acord"
        else -> "Agree"
    }
}

fun load_disagreeButton(): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> "Disagree"
        "ro" -> "Nu sunt de acord"
        else -> "Disagree"
    }
}

fun load_aboutSubtitle(context: Context): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.about_en)
        "ro" -> context.getString(R.string.about_ro)
        else -> context.getString(R.string.about_en)
    }
}

fun load_pitchSpeed(context: Context, case: Boolean): String {
    return if (case) {
        when (AppConfig.mainLanguage.code) {
            "en" -> context.getString(R.string.tts_pitch_en)
            "ro" -> context.getString(R.string.tts_pitch_ro)
            else -> context.getString(R.string.tts_pitch_en)
        }
    } else {
        when (AppConfig.mainLanguage.code) {
            "en" -> context.getString(R.string.tts_speed_en)
            "ro" -> context.getString(R.string.tts_speed_ro)
            else -> context.getString(R.string.tts_speed_en)
        }
    }
}

fun load_infoBB(context: Context): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.bbox_info_en)
        "ro" -> context.getString(R.string.bbox_info_ro)
        else -> context.getString(R.string.bbox_info_en)
    }
}

fun load_infoCaption(context: Context, case: Boolean): String {
    return if (!case) {
        when (AppConfig.mainLanguage.code) {
            "en" -> context.getString(R.string.caption_info_en)
            "ro" -> context.getString(R.string.caption_info_ro)
            else -> context.getString(R.string.caption_info_en)
        }
    } else {
        when (AppConfig.mainLanguage.code) {
            "en" -> context.getString(R.string.ui_size_info_en)
            "ro" -> context.getString(R.string.ui_size_info_ro)
            else -> context.getString(R.string.ui_size_info_en)
        }
    }
}

fun load_hashCacheTitle(context: Context): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.hash_cache_title_en)
        "ro" -> context.getString(R.string.hash_cache_title_ro)
        else -> context.getString(R.string.hash_cache_title_en)
    }
}

fun load_hashCacheInfoFirst(context: Context): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.hash_cache_info_first_en)
        "ro" -> context.getString(R.string.hash_cache_info_first_ro)
        else -> context.getString(R.string.hash_cache_info_first_en)
    }
}

fun load_hashCacheInfoSecond(context: Context): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.hash_cache_info_second_en)
        "ro" -> context.getString(R.string.hash_cache_info_second_ro)
        else -> context.getString(R.string.hash_cache_info_second_en)
    }
}

fun load_envReportsTitle(context: Context): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.env_reports_title_en)
        "ro" -> context.getString(R.string.env_reports_title_ro)
        else -> context.getString(R.string.env_reports_title_en)
    }
}

fun load_envReportsInfo(context: Context): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.env_reports_info_en)
        "ro" -> context.getString(R.string.env_reports_info_ro)
        else -> context.getString(R.string.env_reports_info_en)
    }
}

fun load_sceneClassifierError(context: Context): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.classifier_error_en)
        "ro" -> context.getString(R.string.classifier_error_ro)
        else -> context.getString(R.string.classifier_error_en)
    }
}

fun load_speechRecognizerError(context: Context): String {
    return context.getString(R.string.stt_error_en)
}

fun load_translaterError(context: Context): String {
    return context.getString(R.string.translator_error_ro)
}

fun load_homeTitle(): String {
    return if (AppConfig.mainLanguage.code == "en") {
        "Hello ~${AppConfig.user_name}~,\nwhat can I do for you?"
    } else {
        "Salut ~${AppConfig.user_name}~,\ncu ce te pot ajuta?"
    }
}

fun load_tutorialDialog(context: Context): String {
    return when (PhoneStatusMonitor.getInstance().ttsManager.currentLanguage) {
        "en" -> context.getString(R.string.intro_dialog_en)
        "ro" -> context.getString(R.string.intro_dialog_ro)
        else -> context.getString(R.string.intro_dialog_en)
    }
}

fun load_detectionTutorial(context: Context, step: Int): String {
    return if (AppConfig.mainLanguage.code == "en") {
        when (step) {
            1 -> context.getString(R.string.detect1_en)
            2 -> context.getString(R.string.detect2_en)
            else -> load_homeTitle()
        }
    } else {
        when (step) {
            1 -> context.getString(R.string.detect1_ro)
            2 -> context.getString(R.string.detect2_ro)
            else -> load_homeTitle()
        }
    }
}

fun load_detectionTutorialSpeak(context: Context, step: Int): String {
    return if (PhoneStatusMonitor.getInstance().ttsManager.currentLanguage == "en") {
        when (step) {
            1 -> context.getString(R.string.detect1_en)
            2 -> context.getString(R.string.detect2_en)
            else -> ""
        }
    } else {
        when (step) {
            1 -> context.getString(R.string.detect1_ro)
            2 -> context.getString(R.string.detect2_ro)
            else -> ""
        }
    }
}

fun load_captionTutorial(context: Context, step: Int): String {
    return if (AppConfig.mainLanguage.code == "en") {
        when (step) {
            1 -> context.getString(R.string.caption1_en)
            2 -> context.getString(R.string.caption2_en)
            else -> load_homeTitle()
        }
    } else {
        when (step) {
            1 -> context.getString(R.string.caption1_ro)
            2 -> context.getString(R.string.caption2_ro)
            else -> load_homeTitle()
        }
    }
}

fun load_captionTutorialSpeak(context: Context, step: Int): String {
    return if (PhoneStatusMonitor.getInstance().ttsManager.currentLanguage == "en") {
        when (step) {
            3 -> context.getString(R.string.caption1_en)
            4 -> context.getString(R.string.caption2_en)
            else -> ""
        }
    } else {
        when (step) {
            3 -> context.getString(R.string.caption1_ro)
            4 -> context.getString(R.string.caption2_ro)
            else -> ""
        }
    }
}

fun load_speakTutorial(context: Context, step: Int): String {
    return when (step) {
        1 -> context.getString(R.string.findmyobj1_en)
        2 -> context.getString(R.string.findmyobj2_en)
        3 -> context.getString(R.string.findmyobj3_en)
        4 -> context.getString(R.string.findmyobj4_en)
        5 -> context.getString(R.string.findmyobj5_en)
        6 -> context.getString(R.string.findmyobj6_en)
        7 -> context.getString(R.string.findmyobj7_en)
        else -> load_homeTitle()
    }
}

fun load_speakTutorialSpeak(context: Context, step: Int): String {
    return when (step) {
        5 -> context.getString(R.string.findmyobj1_en)
        6 -> context.getString(R.string.findmyobj2_en)
        7 -> context.getString(R.string.findmyobj3_en)
        8 -> context.getString(R.string.findmyobj4_en)
        9 -> context.getString(R.string.findmyobj5_en)
        10 -> context.getString(R.string.findmyobj6_en)
        11 -> context.getString(R.string.findmyobj7_en)
        else -> ""
    }
}

fun load_syncStatusText(days: Int): String {
    return if (AppConfig.mainLanguage.code == "en") {
        "$days days since last sync"
    } else {
        "$days zile de la ultima sincronizare"
    }
}

fun load_syncStatusTextSpeech(days: Int): String {
    return if (PhoneStatusMonitor.getInstance().ttsManager.currentLanguage == "en") {
        "$days days since last sync"
    } else {
        "$days zile de la ultima sincronizare"
    }
}

fun load_syncErrorText(context: Context): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.sync_error_en)
        "ro" -> context.getString(R.string.sync_error_ro)
        else -> context.getString(R.string.sync_error_en)
    }
}

fun load_syncErrorSpeech(context: Context): String {
    return when (PhoneStatusMonitor.getInstance().ttsManager.currentLanguage) {
        "en" -> context.getString(R.string.sync_error_en)
        "ro" -> context.getString(R.string.sync_error_ro)
        else -> context.getString(R.string.sync_error_en)
    }
}

fun load_homePageIntro(context: Context): String {
    return when (PhoneStatusMonitor.getInstance().ttsManager.currentLanguage) {
        "en" -> context.getString(R.string.home_page_intro_en)
        "ro" -> context.getString(R.string.home_page_intro_ro)
        else -> context.getString(R.string.home_page_intro_en)
    }
}

fun load_navigateToSettings(context: Context): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.navigating_to_settings_en)
        "ro" -> context.getString(R.string.navigating_to_settings_ro)
        else -> context.getString(R.string.navigating_to_settings_en)
    }
}

fun load_unavailableSTT(context: Context): String {
    return when (PhoneStatusMonitor.getInstance().ttsManager.currentLanguage) {
        "en" -> context.getString(R.string.stt_unavailable_en)
        "ro" -> context.getString(R.string.stt_unavailable_ro)
        else -> context.getString(R.string.stt_unavailable_en)
    }
}

fun load_errorSTT(context: Context): String {
    return context.getString(R.string.stt_error_load_en)
}

fun load_errorSTTRuntime(context: Context): String {
    return context.getString(R.string.stt_error_runtime_en)
}

fun load_talkbackError(context: Context): String {
    return when (PhoneStatusMonitor.getInstance().ttsManager.currentLanguage) {
        "en" -> context.getString(R.string.talkback_error_en)
        "ro" -> context.getString(R.string.talkback_error_ro)
        else -> context.getString(R.string.talkback_error_en)
    }
}

fun startBatteryLevelCheck(
    keepRunning: MutableState<Boolean>,
    showWarning: MutableState<Boolean>,
    avgBatteryMoreUsed: MutableState<Float>
) {
    val context = PhoneStatusMonitor.getInstance().currentContext
    var previousTimestamp = System.currentTimeMillis()
    var previousLevel = Utils.returnBatteryLevel(context)
    var previousDiff = 0L

    val batteryReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            if (!keepRunning.value) {
                context?.unregisterReceiver(this)
                return
            }

            val currentTimestamp = System.currentTimeMillis()
            val currentLevel = Utils.returnBatteryLevel(context!!)
            val currentDiff = currentTimestamp - previousTimestamp

            if (previousDiff > 0 && currentLevel < previousLevel) {
                // Calculate consumption rate ratio
                val ratio = currentDiff.toFloat() / previousDiff.toFloat()
                // Update average
                if (ratio < 1.0f) {
                    val currentAvg = avgBatteryMoreUsed.value
                    if (currentAvg == 0f) {
                        avgBatteryMoreUsed.value = ratio
                    } else {
                        avgBatteryMoreUsed.value = (currentAvg + ratio) / 2f
                    }
                }
                if (ratio < Constants.BATTERY_USAGE_THRESHOLD) {
                    // Battery is draining faster than previous interval

                    // Show warning
                    showWarning.value = true

                    // Hide warning after delay
                    CoroutineScope(Dispatchers.Main).launch {
                        delay(Constants.BATTERY_WARNING_DISPLAY_MS)
                        showWarning.value = false
                    }

                    Log.d(
                        "BatteryCheck",
                        "Battery usage increased: ratio=$ratio, avg=${avgBatteryMoreUsed.value}"
                    )
                }
            }

            previousTimestamp = currentTimestamp
            previousLevel = currentLevel
            previousDiff = currentDiff
        }
    }

    // Register receiver
    val filter = IntentFilter(Intent.ACTION_BATTERY_CHANGED)
    context.registerReceiver(batteryReceiver, filter)

    // Monitor keepRunning state in coroutine
    CoroutineScope(Dispatchers.Default).launch {
        while (keepRunning.value) {
            delay(1000)
        }
        // Unregister when stopped
        try {
            context.unregisterReceiver(batteryReceiver)
        } catch (e: Exception) {
            Log.e("BatteryCheck", "Error unregistering receiver", e)
        }
    }
}

fun load_scanningScene(context: Context): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.scanning_scene_en)
        "ro" -> context.getString(R.string.scanning_scene_ro)
        else -> context.getString(R.string.scanning_scene_en)
    }
}

fun load_noObjectsFound(context: Context): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.no_objects_found_en)
        "ro" -> context.getString(R.string.no_objects_found_ro)
        else -> context.getString(R.string.no_objects_found_en)
    }
}

fun load_classificationSuccess(sceneName: String): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> "Image classified successfully: ~$sceneName"
        "ro" -> "Imagine clasificată cu succes: ~$sceneName"
        else -> "Image classified successfully: ~$sceneName"
    }
}

fun load_speakNoObjectsFound(index: Int): String {
    return when (PhoneStatusMonitor.getInstance().ttsManager.currentLanguage) {
        "en" ->
            if (index + 1 > 1)
                "The application detected ${index} objects in your environment"
            else
                "The application detected an single object in your environment"

        "ro" ->
            if (index + 1 > 1)
                "Aplicația a detectat ${index} obiecte în spațiul în care vă aflați"
            else
                "Aplicația a detectat un singur obiect în spațiul în care vă aflați"

        else ->
            if (index + 1 > 1)
                "The application detected ${index} objects in your environment"
            else
                "The application detected an single object in your environment"
    }
}

fun searchHashCache(targetHash: String): List<Int>? {
    try {
        val context = PhoneStatusMonitor.getInstance().currentContext
        val cacheFile = File(FileUtils.getProfileDirectory(context), Constants.HASH_CACHE_FILE_NAME)

        if (cacheFile.length() == 0L) {
            Log.d("HashCache", "Cache file empty or not found")
            return null
        }

        // Read all records
        val records = cacheFile.readLines()
        val totalRecords = records.size

        if (totalRecords == 0) return null

        Log.d("HashCache", "Searching through $totalRecords records...")

        // Calculate number of threads (5 threads if > 10 records, else 1)
        val threadCount = if (totalRecords <= 10) 1 else 5
        val recordsPerThread = totalRecords / threadCount

        Log.d("HashCache", "Using $threadCount threads/$recordsPerThread records per thread")

        // Shared state
        val foundResult = AtomicBoolean(false)
        var resultTokens: List<Int>? = null
        val finishedThreads = AtomicInteger(0)
        val latch = CountDownLatch(threadCount)

        // Launch threads
        for (threadId in 0 until threadCount) {
            val startIdx = threadId * recordsPerThread
            val endIdx =
                if (threadId == threadCount - 1) totalRecords else (threadId + 1) * recordsPerThread

            Thread {
                try {
                    Log.d("HashCache", "Thread $threadId: scanning records $startIdx to $endIdx")

                    for (i in startIdx until endIdx) {
                        // Check if another thread found result
                        if (foundResult.get()) {
                            Log.d(
                                "HashCache",
                                "Thread $threadId: stopping (result found by another thread)"
                            )
                            break
                        }

                        val record = records[i].trim()
                        if (record.isEmpty()) continue

                        // Parse record: "hash:tokenId0_tokenId1_tokenId2"
                        val parts = record.split(":", limit = 2)
                        if (parts.size != 2) continue

                        val recordHash = parts[0]
                        val tokensPart = parts[1]

                        if (similarity(targetHash, recordHash)) {
                            // Found match!
                            Log.d("HashCache", "Thread $threadId: FOUND SIMILARITY!")

                            // Try to set result (only first thread succeeds)
                            if (foundResult.compareAndSet(false, true)) {
                                // Parse tokens
                                resultTokens = tokensPart.split("_").mapNotNull { it.toIntOrNull() }
                                Log.d("HashCache", "Result set: ${resultTokens.size} tokens")
                                latch.countDown()
                            }
                            break
                        }
                    }
                } catch (e: Exception) {
                    Log.e("HashCache", "Thread $threadId error", e)
                } finally {
                    finishedThreads.incrementAndGet()
                    if (finishedThreads.get() == threadCount)
                        latch.countDown()
                }
            }.start()
        }

        // Wait for threads to finish (max 5 seconds timeout)
        val finished = latch.await(Constants.MAX_WAIT_SIMILARITY.toLong(), TimeUnit.MILLISECONDS)

        if (!finished) {
            Log.w("HashCache", "Search timeout after 5 seconds")
        }

        Log.d(
            "HashCache",
            "Search complete: ${finishedThreads.get()}/$threadCount threads finished, found: ${foundResult.get()}"
        )

        return resultTokens

    } catch (e: Exception) {
        Log.e("HashCache", "Error searching cache", e)
        return null
    }
}

fun saveToHashCache(hash: String, tokenIds: List<Int>) {
    BackgroundTaskExecutor.getInstance().executeAsync(
        {
            try {
                val context = PhoneStatusMonitor.getInstance().currentContext
                val cacheFile =
                    File(FileUtils.getProfileDirectory(context), Constants.HASH_CACHE_FILE_NAME)

                // Determine max records based on env_reports mode
                val maxRecords =
                    if (AppConfig.hash_caching.equals("heavy")) Constants.HC_MAX_RECORDS_HEAVY else Constants.HC_MAX_RECORDS_LIGHT
                val removeCount = (maxRecords * 0.40).toInt()

                // Read existing records
                val existingRecords = if (cacheFile.exists()) {
                    cacheFile.readLines().toMutableList()
                } else {
                    mutableListOf()
                }

                // Create new record: "hash:tokenId0_tokenId1_tokenId2"
                val tokenString = tokenIds.joinToString("_")
                val newRecord = "$hash:$tokenString"

                // Check if file is too large
                if (existingRecords.size >= maxRecords) {
                    Log.d(
                        "HashCache",
                        "Cache full (${existingRecords.size}/$maxRecords), removing $removeCount old records"
                    )
                    // Remove oldest records (from beginning)
                    repeat(removeCount) {
                        if (existingRecords.isNotEmpty()) {
                            existingRecords.removeAt(0)
                        }
                    }
                }

                // Add new record
                existingRecords.add(newRecord)

                // Write back to file
                BufferedWriter(FileWriter(cacheFile)).use { writer ->
                    existingRecords.forEach { record ->
                        writer.write(record)
                        writer.newLine()
                    }
                }

                Log.d("HashCache", "Saved to cache: ${existingRecords.size} total records")
            } catch (e: Exception) {
                Log.e("HashCache", "Error saving to cache", e)
            }
        },
        {
            PhoneStatusMonitor.getInstance().writingToHCFinished = true
        },
        {
            PhoneStatusMonitor.getInstance().writingToHCFinished = true
        })
}

fun load_captioningScene(context: Context): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.captioning_scene_en)
        "ro" -> context.getString(R.string.captioning_scene_ro)
        else -> context.getString(R.string.captioning_scene_en)
    }
}

fun load_captionError(context: Context): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.caption_error_en)
        "ro" -> context.getString(R.string.caption_error_ro)
        else -> context.getString(R.string.caption_error_en)
    }
}

fun load_captionError2(context: Context): String {
    return when (PhoneStatusMonitor.getInstance().ttsManager.currentLocale.language) {
        "en" -> context.getString(R.string.blind_caption_error_en)
        "ro" -> context.getString(R.string.blind_caption_error_ro)
        else -> context.getString(R.string.blind_caption_error_en)
    }
}

fun load_loadingReports(context: Context): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.making_reports_en)
        "ro" -> context.getString(R.string.making_reports_ro)
        else -> context.getString(R.string.making_reports_en)
    }
}