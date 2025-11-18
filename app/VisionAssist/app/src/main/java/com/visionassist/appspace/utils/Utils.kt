package com.visionassist.appspace.utils

import android.content.Context
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import com.visionassist.appspace.R

data class Language(
    var code: String,
    var name: String,
    var country: String
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

fun load_errorTextBlind(context: Context,exitCode: Int): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.exit_error2_en)+exitCode.toString()
        "ro" -> context.getString(R.string.exit_error2_ro)+exitCode.toString()
        else -> context.getString(R.string.exit_error2_en)+exitCode.toString()
    }
}

fun haptic_model0(): LongArray {
    return longArrayOf(0, 250)
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

fun load_infoLoadProfileActivity(context: Context): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> "Please select the '${Constants.PROFILE_FOLDER_NAME}' folder from your storage to load your profile"
        "ro" -> "Vă rugăm să selectați folderul '${Constants.PROFILE_FOLDER_NAME}' din fișierele dvs. pentru a încărca profilul"
        else -> "Please select the '${Constants.PROFILE_FOLDER_NAME}' folder from your storage to load your profile"
    }
}

fun load_errorLocalLoadProfileActivity(context: Context,errorCode: Int): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> "Error was encountered while fetching the profile\n\n@(Error code: ${errorCode})"
        "ro" -> "A apărut o eroare în timpul încărcării profilului dvs.\n\n@(Error code: ${errorCode})"
        else -> "Error was encountered while fetching the profile\n\n@(Error code: ${errorCode})"
    }
}

fun load_successLocalLoadProfileActivity(context: Context,errorCode: Int): String {
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

fun load_passChangedSuccess(context: Context,emailInput: String): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> "Password change request sent to\n${emailInput}"
        "ro" -> "Cerere de schimbare a parolei trimisă către\n${emailInput}"
        else -> "Password change request sent to\n${emailInput}"
    }
}

fun load_profileImportedSuccess(context: Context): String {
    return  when (AppConfig.mainLanguage.code) {
        "en" -> "Profile imported successfully"
        "ro" -> "Profilul a fost importat cu succes"
        else -> "Profile imported successfully"
    }
}

fun load_emailAlreadyExists(context: Context): String {
    return  when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.already_exists_account_en)
        "ro" -> context.getString(R.string.already_exists_account_ro)
        else -> context.getString(R.string.already_exists_account_en)
    }
}

fun load_invalidEmail(context: Context): String {
    return  when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.invalid_email_en)
        "ro" -> context.getString(R.string.invalid_email_ro)
        else -> context.getString(R.string.invalid_email_en)
    }
}

fun load_genericErrorNew(context: Context): String {
    return  when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.generic_error2_en)
        "ro" -> context.getString(R.string.generic_error2_ro)
        else -> context.getString(R.string.generic_error2_en)
    }
}

fun load_invalidCombination(context: Context): String {
    return  when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.invalid_chars_en)
        "ro" -> context.getString(R.string.invalid_chars_ro)
        else -> context.getString(R.string.invalid_chars_en)
    }
}

fun load_contributeResearch(context: Context): String {
    return  when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.contribute_en)
        "ro" -> context.getString(R.string.contribute_ro)
        else -> context.getString(R.string.contribute_en)
    }
}

fun load_whatsYourName(context: Context): String {
    return  when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.hello_en)
        "ro" -> context.getString(R.string.hello_ro)
        else -> context.getString(R.string.hello_en)
    }
}

fun load_howOldAreYou(context: Context): String {
    return  when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.old_en)
        "ro" -> context.getString(R.string.old_ro)
        else -> context.getString(R.string.old_en)
    }
}

fun load_whatTypeOfVision(context: Context): String {
    return  when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.visual_condition_en)
        "ro" -> context.getString(R.string.visual_condition_ro)
        else -> context.getString(R.string.visual_condition_en)
    }
}

fun load_agreeButton(context: Context): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> "Agree"
        "ro" -> "De acord"
        else -> "Agree"
    }
}

fun load_disagreeButton(context: Context): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> "Disagree"
        "ro" -> "Nu sunt de acord"
        else -> "Disagree"
    }
}

fun load_aboutSubtitle(context: Context): String {
    return  when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.about_en)
        "ro" -> context.getString(R.string.about_ro)
        else -> context.getString(R.string.about_en)
    }
}

fun load_pitchSpeed(context: Context,case: Boolean): String {
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
    return  when (AppConfig.mainLanguage.code) {
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

fun load_homeTitle(context: android.content.Context): String {
    return if (com.visionassist.appspace.utils.AppConfig.mainLanguage.code == "en") {
        "Hello ~${com.visionassist.appspace.utils.AppConfig.user_name}, what can I do for you?"
    } else {
        "Salut ~${com.visionassist.appspace.utils.AppConfig.user_name}, cu ce te pot ajuta?"
    }
}

fun load_detectionTutorial(context: android.content.Context, step: Int): String {
    return if (com.visionassist.appspace.utils.AppConfig.mainLanguage.code == "en") {
        when (step) {
            1 -> "Volume up button ~goes into detection activity"
            2 -> "The activity could be ~shortcut on lock screen with an icon, you can turn this on from settings"
            else -> load_homeTitle(context)
        }
    } else {
        when (step) {
            1 -> "Butonul de volum sus ~intră în activitatea de detectare"
            2 -> "Activitatea poate fi ~scurtată pe ecranul de blocare cu o pictogramă, poți activa acest lucru din setări"
            else -> load_homeTitle(context)
        }
    }
}

fun load_captionTutorial(context: android.content.Context, step: Int): String {
    return if (com.visionassist.appspace.utils.AppConfig.mainLanguage.code == "en") {
        when (step) {
            1 -> "Volume down button ~activates caption"
            2 -> "The activity could be ~shortcut from settings"
            else -> load_homeTitle(context)
        }
    } else {
        when (step) {
            1 -> "Butonul de volum jos ~activează caption"
            2 -> "Activitatea poate fi ~scurtată din setări"
            else -> load_homeTitle(context)
        }
    }
}

fun load_speakTutorial(context: android.content.Context, step: Int): String {
    if (com.visionassist.appspace.utils.AppConfig.mainLanguage.code != "en") {
        return load_homeTitle(context)
    }

    return when (step) {
        1 -> "Rapidly press ~volume down button twice to start speaking"
        2 -> "When you speak ~words will be prompted on screen in white"
        3 -> "Press volume down ~to cancel speaking"
        4 -> "Press volume down ~to approve what the model recorded"
        5 -> "Press volume up ~to retry speaking"
        6 -> "Press anywhere on screen ~to cancel"
        else -> load_homeTitle(context)
    }
}

fun load_syncStatusText(context: android.content.Context, days: Int): String {
    return if (com.visionassist.appspace.utils.AppConfig.mainLanguage.code == "en") {
        "$days days since last sync"
    } else {
        "$days zile de la ultima sincronizare"
    }
}

fun load_syncErrorText(context: android.content.Context): String {
    return if (com.visionassist.appspace.utils.AppConfig.mainLanguage.code == "en") {
        "Sync failed ~Make sure you have network access and restart the app to solve this"
    } else {
        "Sincronizarea a eșuat ~Asigură-te că ai acces la rețea și repornește aplicația pentru a rezolva acest lucru"
    }
}