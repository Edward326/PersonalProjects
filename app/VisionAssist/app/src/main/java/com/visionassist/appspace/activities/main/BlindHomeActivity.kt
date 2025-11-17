//cand se apasa volum jos,sus caption/deetion
//de doua ori apasat volumul jos, voice to speech pentrua  cauta obiectul
//detection sau caption verifica, peromsiunile la camera, dupa checkPhoneStatus,
/*
 facut metoda prin care:
                -se separa fieacrer word din setcene de la voice to text model, si se elimina cuvintele inutile(where is, etc),
                -pt cuvintele ramase se cauta sinnomie din tabelul cu sinonime, pentru cele care se gasesc se incluisesc cu idul clasei yolo in secv finala, daca nu se gaseste nu se pune in secv finala
                -se ia secv finala si se de la yolo, iar cand gaseste un obiect din secv finala,

            *facut activitaeta pentru vazatorri, este live, cand obtine clasele, daca se gasesc obiectele din secv finala(se elimina din lista), se opreste activtatea, si afiseaza pe ecran imaginea cu ibiectele cu bboxuri,
            de aici daca mai sunt nuy mai sunt elem ramse in secv finala afiseaz doar butonul de home, dar daca mai sunt afiseaza butonul de home si next(repeta aceaasi activiate pana cand se gasesc sau parasete userul prin home)
            (in live butonul de volum daca este apasa cel de jos iese din activtate)

            *facut activitatea pentru nevazatori, este live, cand obtine clasele, daca se gasesc obiectele din secv finala(se elimina din lista), se opreste activtatea, se pune imaginea care in care s-au gasit si face speech cu obiectele gasite,
             de aici daca mai sunt nuy mai sunt elem ramse in secv finala afiseaz doar butonul de home, dar daca mai sunt afiseaza butonul de home si next(repeta aceaasi activiate pana cand se gasesc sau parasete userul prin home)
            (in live butonul de volum daca este apasa cel de jos iese din activtate)
 */
package com.visionassist.appspace.activities.main

import android.os.Bundle
import android.util.Log
import android.view.KeyEvent
import android.view.View
import android.widget.ImageView
import androidx.appcompat.app.AppCompatActivity
import com.visionassist.appspace.ExceptionVisionAssist
import com.visionassist.appspace.PhoneStatusMonitor
import com.visionassist.appspace.R
import com.visionassist.appspace.jetpack.managers.ErrorDialogManager
import com.visionassist.appspace.utils.AppConfig
import com.visionassist.appspace.utils.Constants
import com.visionassist.appspace.utils.FileUtils
import java.io.IOException

class BlindHomeActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContentView(R.layout.activity_main)
        val logoImage = findViewById<ImageView?>(R.id.logo_image)
        logoImage.setVisibility(View.VISIBLE)

        try {
            val content1 = "Data of the hash_cache.json:\n" + FileUtils.loadFileAsString(
                FileUtils.getHashCacheInputStream(this)
            )
            //String content2 = "\n\nData of the env_reports.json:\n"+FileUtils.loadFileAsString(FileUtils.getEnvReportsInputStream(this));
            val content3 = "\n\nData of the AppConfig:\n" + AppConfig.listAppConfig()
            val content4 =
                "\n\nIsInitProfileLoaded:\t" + PhoneStatusMonitor.getInstance().profileLoaded
            val content5 = "\n\nData of the profile.json:\n" + FileUtils.loadFileAsString(
                FileUtils.getProfileInputStream(this)
            )

            val concat = content1 + content3 + content4 + content5
            Log.d(TAG, "BlindHomeActivity created\n\n---STATUS---\n" + concat)
        } catch (e: IOException) {
            handleProfileError(e)
        }
    }

    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        // Use a switch statement for key code checks
        return when (keyCode) {
            KeyEvent.KEYCODE_VOLUME_DOWN -> {
                Log.d(TAG, "Volume button down pressed")
                true
            }

            KeyEvent.KEYCODE_VOLUME_UP -> {
                Log.d(TAG, "Volume button up pressed")
                true
            }

            else ->  // For all other keys, call the super implementation
                super.onKeyDown(keyCode, event)
        }
    }

    private fun handleProfileError(e: Exception?) {
        val monitor = PhoneStatusMonitor.getInstance()
        if (e is ExceptionVisionAssist) {
            val ref = e.getLoadingManager()
            val errorCode = e.getErrorCode()

            Log.e(TAG, "Thrown special exception, error code: " + errorCode)

            val errorDialog = ErrorDialogManager(monitor.getCurrentActivity())
            errorDialog.setupDialog(errorCode)
            if (ref != null) ref.hideLoading()
            monitor.shutdownApp(errorDialog, monitor.getCurrentContext())
        } else {
            Log.e(TAG, "Thrown exception, explanation: ", e)
            val errorDialog = ErrorDialogManager(monitor.getCurrentActivity())
            errorDialog.setupDialog(Constants.EXCEPTION_CLASS_ERROR)
            monitor.shutdownApp(errorDialog, monitor.getCurrentContext())
        }
    }

    companion object {
        private const val TAG = "BlindHomeActivity"
    }
}