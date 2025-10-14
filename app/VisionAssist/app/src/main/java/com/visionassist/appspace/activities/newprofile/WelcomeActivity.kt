package com.visionassist.appspace.activities.newprofile

import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.compose.ui.platform.ComposeView
import com.visionassist.appspace.PhoneStatusMonitor
import com.visionassist.appspace.R
import com.visionassist.appspace.jetpack.design.LanguageSelector
import com.visionassist.appspace.jetpack.design.NavigationButtons
import com.visionassist.appspace.jetpack.managers.ErrorDialogManager
import com.visionassist.appspace.jetpack.managers.LoadingManager
import com.visionassist.appspace.utils.AppConfig
import com.visionassist.appspace.utils.Constants
import com.visionassist.appspace.utils.FileUtils
import com.visionassist.appspace.utils.Language
import org.json.JSONObject

class WelcomeActivity : AppCompatActivity() {
    private val TAG = "WelcomeActivity"

    private var selectedLanguage = Language("en", "English", "US")
    private lateinit var loadingManager: LoadingManager

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_welcome)

        // Initialize views
        val logoImage = findViewById<ImageView>(R.id.logo_image)
        val welcomeText = findViewById<TextView>(R.id.welcome_text)
        val appNameText = findViewById<TextView>(R.id.app_name_text)
        val languageSelector = findViewById<ComposeView>(R.id.language_selector)
        val nextButton = findViewById<ComposeView>(R.id.next_button)
        val loadingBox = findViewById<ComposeView>(R.id.loading_box)
        loadingManager = LoadingManager(loadingBox, false, this)
        loadingManager.setupLoadingBox()

        logoImage.visibility = View.VISIBLE
        welcomeText.visibility = View.VISIBLE
        appNameText.visibility = View.VISIBLE

        languageSelector.setContent {
            LanguageSelector { language ->
                selectedLanguage = language
                Log.d(TAG, "Language selected: ${language.name} (${language.code})")
            }
        }

        nextButton.setContent {
            NavigationButtons(
                onBackClick = ::onBackPressedMod,
                onNextClick = ::onNextPressedMod
            )
        }
    }

    private fun onBackPressedMod() {
        val intent = Intent(this, ConfigurationActivity::class.java)
        startActivity(intent)
        finish() // Remove from activity stack
    }

    private fun onNextPressedMod() {
        writeLanguageToProfile()

        AppConfig.mainLanguage = selectedLanguage

        val intent = Intent(this, LoadingActivity::class.java)
        startActivity(intent)
        finish()
    }

    private fun writeLanguageToProfile() {
        try {
            // Read existing profile
            val jsonObject: JSONObject = if (FileUtils.profileFileExists(this)) {
                val content = FileUtils.loadFileAsString(FileUtils.getProfileInputStream(this))
                JSONObject(content)
            } else {
                JSONObject()
            }

            jsonObject.put("language_code", selectedLanguage.code)
            jsonObject.put("language_desc", selectedLanguage.name)
            jsonObject.put("language_country", selectedLanguage.country)

            // Write back to file
            FileUtils.writeProfileFile(this, this, jsonObject.toString())

            Log.d(TAG, "WelcomeActivity has written to the profile file")

        } catch (e: Exception) {
            val phoneMonitor = PhoneStatusMonitor.getInstance()
            val errorDialog = ErrorDialogManager(this)
            errorDialog.setupDialog(Constants.FILE_WRITE_ERROR, getString(R.string.exit_error_en))
            phoneMonitor.shutdownApp(errorDialog, this)
        }
    }
}