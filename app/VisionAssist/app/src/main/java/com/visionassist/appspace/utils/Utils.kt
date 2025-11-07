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
    Font(R.font.roboto_regular_ttf)
)

val robotoSemibold = FontFamily(
    Font(R.font.roboto_semibold)
)

val robotoLight = FontFamily(
    Font(R.font.roboto_light_ttf)
)

val robotoExtraBold = FontFamily(
    Font(R.font.roboto_extrabold)
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

fun load_micDeniedInfo(context: Context): String {
    return when (AppConfig.mainLanguage.code) {
        "en" -> context.getString(R.string.microphone_denied_info_en)
        "ro" -> context.getString(R.string.microphone_denied_info_ro)
        else -> context.getString(R.string.microphone_denied_info_en)
    }
}

fun load_micDeniedInfoButtons(context: Context, case: Boolean): String {
    return if (case) {
        when (AppConfig.mainLanguage.code) {
            "en" -> context.getString(R.string.give_access_en)
            "ro" -> context.getString(R.string.give_access_ro)
            else -> context.getString(R.string.give_access_en)
        }
    } else {
        when (AppConfig.mainLanguage.code) {
            "en" -> context.getString(R.string.dont_give_en)
            "ro" -> context.getString(R.string.dont_give_ro)
            else -> context.getString(R.string.dont_give_en)
        }
    }
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

fun haptic_model0(): LongArray {
    return longArrayOf(0, 500)
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