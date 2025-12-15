@file:Suppress("COMPOSE_APPLIER_CALL_MISMATCH")

package com.visionassist.appspace.activities.tabs.reports

import android.annotation.SuppressLint
import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.KeyEvent
import androidx.activity.compose.setContent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.animateScrollBy
import androidx.compose.foundation.gestures.detectHorizontalDragGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.Menu
import androidx.compose.material.icons.filled.MoreHoriz
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.colorResource
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.visionassist.appspace.BaseActivity
import com.visionassist.appspace.R
import com.visionassist.appspace.activities.main.BottomNavigationBar
import com.visionassist.appspace.activities.main.HomeActivity
import com.visionassist.appspace.activities.newprofile.LoadProfileActivity
import com.visionassist.appspace.activities.tabs.settings.SettingsActivity
import com.visionassist.appspace.jetpack.design.LoadingComponent
import com.visionassist.appspace.jetpack.design.NotificationDialog
import com.visionassist.appspace.jetpack.managers.InfoNotificationManager
import com.visionassist.appspace.utils.AppConfig
import com.visionassist.appspace.utils.Constants
import com.visionassist.appspace.utils.haptic_model0
import com.visionassist.appspace.utils.load_loadingReports
import com.visionassist.appspace.utils.load_loadingReportsError
import com.visionassist.appspace.utils.load_reportsEmpty
import com.visionassist.appspace.utils.robotoExtraBold
import com.visionassist.appspace.utils.vibrate
import kotlinx.coroutines.launch

class EnvironmentReportsActivity : BaseActivity() {
    private val TAG = "EnvReportsActivity"

    // Main handler
    private val mainHandler = Handler(Looper.getMainLooper())

    // UI states
    private val showLoading = mutableStateOf(true)
    private val loadingText = mutableStateOf("")
    private val showResults = mutableStateOf(false)
    private val showErrorDialog = mutableStateOf(false)
    private val errorMessage = mutableStateOf("")

    // Statistics data
    private var statistics: EnvironmentStatistics? = null
    private val selectedSceneIndex = mutableStateOf<Int?>(null)

    // Destroyed flag
    private var isDestroyed = false

    private val infoNotificationManager = InfoNotificationManager(this)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        loadingText.value = load_loadingReports(this)

        setContent {
            EnvironmentReportsScreen(
                showLoading = showLoading.value,
                loadingText = loadingText.value,
                showErrorDialog = showErrorDialog.value,
                errorMessage = errorMessage.value,
                showResults = showResults.value,
                statistics = statistics,
                selectedSceneIndex = selectedSceneIndex.value,
                onSceneSelected = ::handleSceneSelection,
                onNavigateHome = ::handleNavigateHome,
                onNavigateReports = ::handleNavigateReports,
                onNavigateSettings = ::handleNavigateSettings,
                onErrorRetry = { handleRetry() },
                onErrorOk = {
                    showErrorDialog.value = false
                    handleNavigateHome()
                }
            )
        }

        mainHandler.postDelayed({
            processReports()
        }, 1000)
    }

    private fun handleRetry() {
        vibrateIfEnabled()

        showErrorDialog.value = false
        showLoading.value = true
        processReports()
    }

    private fun processReports() {
        EnvironmentReportsManagerKt.processReportsAsync(
            context = this,
            onSuccess = { stats ->
                if (isDestroyed) return@processReportsAsync

                statistics = stats
                showLoading.value = false

                // Delay before showing results (like CaptionActivity)
                mainHandler.postDelayed({
                    if (isDestroyed) return@postDelayed

                    if (statistics!!.allEmpty())
                        infoNotificationManager.showNotification(
                            load_reportsEmpty(this),
                            { handleNavigateHome() },
                            "OK"
                        )
                    else
                        showResults.value = true
                }, Constants.ANIMATION_DELAY.toLong())

            }, onError = { e ->
                if (isDestroyed) return@processReportsAsync

                Log.e(TAG, "Error processing reports", e)
                // Show empty statistics
                statistics = EnvironmentStatistics()
                showLoading.value = false

                mainHandler.postDelayed({
                    if (isDestroyed) return@postDelayed
                    errorMessage.value = load_loadingReportsError(this)
                    showErrorDialog.value = true
                }, Constants.ANIMATION_DELAY.toLong())
            })
    }

    private fun vibrateIfEnabled() {
        if (AppConfig.haptics) {
            vibrate(haptic_model0())
        }
    }

    private fun handleNavigateHome() {
        vibrateIfEnabled()

        val intent = Intent(this, HomeActivity::class.java)
        startActivity(intent)
        finish()
    }

    private fun handleNavigateReports() {
        vibrateIfEnabled()
    }

    private fun handleNavigateSettings() {
        vibrateIfEnabled()

        val intent = Intent(this, SettingsActivity::class.java)
        startActivity(intent)
        finish()
    }

    private fun handleSceneSelection(index: Int?) {
        selectedSceneIndex.value = index
    }

    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        when (keyCode) {
            KeyEvent.KEYCODE_VOLUME_DOWN -> {
                Log.d(TAG, "Volume button down pressed")
                return true
            }

            KeyEvent.KEYCODE_VOLUME_UP -> {
                Log.d(TAG, "Volume button up pressed")
                return true
            }
        }
        return super.onKeyDown(keyCode, event)
    }

    override fun onDestroy() {
        super.onDestroy()
        isDestroyed = true
        mainHandler.removeCallbacksAndMessages(null)
    }
}

@Composable
fun EnvironmentReportsScreen(
    showLoading: Boolean,
    loadingText: String,
    showErrorDialog: Boolean,
    errorMessage: String,
    showResults: Boolean,
    statistics: EnvironmentStatistics?,
    selectedSceneIndex: Int?,
    onSceneSelected: (Int?) -> Unit,
    onNavigateHome: () -> Unit,
    onNavigateReports: () -> Unit,
    onNavigateSettings: () -> Unit,
    onErrorRetry: () -> Unit,
    onErrorOk: () -> Unit
) {
    var showStatsPanel by remember { mutableStateOf(false) }

    BoxWithConstraints(
        modifier = Modifier.fillMaxSize()
    ) {
        val screenHeight = maxHeight
        val navbarHeight = 90.dp / maxHeight
        val sectionMain = 1.0f - navbarHeight

        // Background image
        Image(
            painter = painterResource(R.drawable.app_background2),
            contentDescription = null,
            modifier = Modifier.fillMaxSize(),
            contentScale = ContentScale.Crop
        )

        // Loading overlay (fade in instantly, no fade out)
        LoadingComponent(
            isVisible = showLoading, loadingText = loadingText, animSpec = Pair(
                fadeIn(
                    initialAlpha = 0f, animationSpec = tween(durationMillis = 0)  // Instant enter
                ), fadeOut(
                    targetAlpha = 0f,
                    animationSpec = tween(durationMillis = Constants.ANIMATION_DELAY)  // Fade out
                )
            )
        )

        NotificationDialog(
            isVisible = showErrorDialog,
            type = LoadProfileActivity.NotificationType.ERROR,
            message = errorMessage,
            showTwoButtons = true,
            firstButtonLabel = if (AppConfig.mainLanguage.code == "en") "Retry" else "Reîncearcă",
            secondButtonLabel = if (AppConfig.mainLanguage.code == "en") "OK" else "OK",
            firstButtonClick = onErrorRetry,
            secondButtonClick = onErrorOk
        )

        if (showResults && statistics != null) {
            // Background image
            Image(
                painter = painterResource(R.drawable.app_background),
                contentDescription = null,
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Crop
            )

            Column {
                Box(modifier = Modifier.height(screenHeight * 0.045f))

                // Logo
                Image(
                    painter = painterResource(R.drawable.vision_assist_logo),
                    contentDescription = "app logo",
                    modifier = Modifier.size(Constants.LOGO_SIZE.dp)
                )
            }

            Box(
                modifier = Modifier
                    .fillMaxSize()
            ) {
                // Main content
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .fillMaxHeight(sectionMain)
                ) {
                    Column(
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(top = screenHeight * 0.17f),
                        verticalArrangement = Arrangement.SpaceBetween,
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        // Tab 1: Most common scenes
                        ScrollableTab(
                            title = if (AppConfig.mainLanguage.code == "en")
                                "Most common scenes"
                            else
                                "Spații întâlnite des",
                            items = statistics.scenesList.map { it.first to it.second },
                            shape = RoundedCornerShape(
                                topStart = 32.dp,
                                topEnd = 32.dp,
                                bottomStart = 16.dp,
                                bottomEnd = 16.dp
                            )
                        )

                        // Tab 2: Most common objects
                        ScrollableTab(
                            title = if (AppConfig.mainLanguage.code == "en")
                                "Most common objects"
                            else
                                "Obiecte întâlnite des",
                            items = statistics.objectsList.map { it.first to it.second },
                            shape = RoundedCornerShape(16.dp)
                        )

                        // Tab 3: Objects by scene
                        ObjectsBySceneTab(
                            statistics = statistics,
                            selectedSceneIndex = selectedSceneIndex,
                            onSceneSelected = onSceneSelected,
                            shape = RoundedCornerShape(
                                topStart = 16.dp,
                                topEnd = 16.dp,
                                bottomStart = 32.dp,
                                bottomEnd = 32.dp
                            )
                        )

                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(bottom = 16.dp),
                            contentAlignment = Alignment.Center
                        ) {
                            FloatingActionButton(
                                onClick = { showStatsPanel = !showStatsPanel },
                                shape = CircleShape,
                                containerColor = colorResource(R.color.std_purple),
                                contentColor = Color.White,
                                modifier = Modifier.size(55.dp)
                            ) {
                                Icon(
                                    imageVector = Icons.Filled.MoreHoriz,
                                    contentDescription = "More statistics",
                                    modifier = Modifier.size(36.dp)
                                )
                            }
                        }
                    }
                }

                // Bottom Navigation Bar
                Box(
                    modifier = Modifier
                        .align(Alignment.BottomCenter)
                        .fillMaxWidth()
                        .fillMaxHeight(navbarHeight),
                ) {
                    BottomNavigationBar(
                        onNavigateHome = onNavigateHome,
                        onNavigateReports = onNavigateReports,
                        onNavigateSettings = onNavigateSettings,
                        showReports = AppConfig.env_reports,
                        selected=1
                    )
                }

                AnimatedVisibility(
                    visible = showStatsPanel,
                    modifier = Modifier
                        .align(Alignment.BottomCenter)
                        .fillMaxWidth(),
                    enter = slideInVertically(
                        initialOffsetY = { it }  // Start from bottom (off screen)
                    ),
                    exit = slideOutVertically(
                        targetOffsetY = { it } // Slide to bottom (off screen)
                    )
                ) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                    ) {
                        StatsPanel(
                            avgThreads = statistics.avgNoThreads,
                            avgDetectorLatency = statistics.avgDetectorLatency,
                            avgClassifierLatency = statistics.avgClassifierLatency,
                            avgBatteryUsage = statistics.avgPercMoreBatteryUsed,
                            onClose = { showStatsPanel = false })
                    }
                }
            }
        }
    }
}

@Composable
fun ScrollableTab(
    title: String, items: List<Pair<String, Int>>,  // (name, count)
    shape: RoundedCornerShape
) {
    if (items.isEmpty()) return

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 40.dp)
            .clip(shape)
            .background(Color(0x66808080))  // Semi-transparent gray
            .padding(20.dp)
    ) {
        // Title box
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(32.dp))
                .background(Color(0xFFFFDCE7))
                .padding(vertical = 12.dp),
            contentAlignment = Alignment.Center
        ) {
            Text(
                text = title,
                fontSize = Constants.STD_ERROR_FONT_SIZE.sp,
                color = Color.Black,
                fontFamily = robotoExtraBold,
                textAlign = TextAlign.Center
            )
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Scrollable items (3 visible at a time)
        ScrollableItemRow(items = items, sum = items.sumOf { it.second })
    }
}

@Composable
fun ScrollableItemRow(items: List<Pair<String, Int>>, sum: Int) {
    val listState = rememberLazyListState()
    val coroutineScope = rememberCoroutineScope()

    LazyRow(
        state = listState,
        modifier = Modifier
            .fillMaxWidth()
            .pointerInput(Unit) {
                detectHorizontalDragGestures(onHorizontalDrag = { _, dragAmount ->
                    coroutineScope.launch {
                        listState.animateScrollBy(dragAmount)
                    }
                })
            },
        horizontalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        itemsIndexed(items) { index, (name, count) ->
            ItemCard(
                name = name,
                percentage = (count.toFloat() / sum * 100).toInt(),
                color = if (index > 2) Color(0xFF625B71) else colorResource(R.color.std_purple)
            )
        }
    }
}

@Composable
fun ItemCard(name: String, percentage: Int, color: Color) {
    Row(
        modifier = Modifier
            .clip(RoundedCornerShape(32.dp))
            .background(Color.White)
            .padding(horizontal = 7.dp, vertical = 7.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Box(
            modifier = Modifier
                .size(35.dp)
                .clip(RoundedCornerShape(16.dp))
                .background(color),
            contentAlignment = Alignment.Center
        ) {
            Text(
                text = "${percentage}%",
                fontSize = 12.sp,
                color = Color(0xFFFFC4D6),
                fontFamily = robotoExtraBold,
                maxLines = 1
            )
        }
        Text(
            text = name,
            fontSize = Constants.STD_FONT_SIZE.sp,
            color = colorResource(R.color.std_purple_dark),
            fontFamily = robotoExtraBold,
            maxLines = 1
        )
    }
}

@Composable
fun ObjectsBySceneTab(
    statistics: EnvironmentStatistics,
    selectedSceneIndex: Int?,
    onSceneSelected: (Int?) -> Unit,
    shape: RoundedCornerShape
) {
    if (statistics.objectsBySceneList.isEmpty()) return

    var showSceneDropdown by remember { mutableStateOf(false) }

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 40.dp)
            .clip(shape)
            .background(Color(0x66808080))
            .padding(20.dp)
    ) {
        // Title box
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(32.dp))
                .background(Color(0xFFFFDCE7))
                .padding(vertical = 12.dp),
            contentAlignment = Alignment.Center
        ) {
            Text(
                text = if (AppConfig.mainLanguage.code == "en")
                    "Most common objects by scene"
                else
                    "Obiecte întâlnite des, in diferite spații",
                fontSize = Constants.STD_ERROR_FONT_SIZE.sp,
                color = Color.Black,
                fontFamily = robotoExtraBold,
                textAlign = TextAlign.Center
            )
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Scene selector
        Box {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(16.dp))
                    .background(Color.White)
                    .clickable { showSceneDropdown = !showSceneDropdown }
                    .padding(12.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Icon(
                        imageVector = Icons.Filled.Menu,
                        contentDescription = null,
                        tint = Color.Gray
                    )
                    Text(
                        text = if (selectedSceneIndex != null && selectedSceneIndex < statistics.scenesList.size) {
                            statistics.scenesList[selectedSceneIndex].first
                        } else {
                            if (AppConfig.mainLanguage.code == "en")
                                "Select a scene"
                            else
                                "Selectează un spațiu"
                        }, fontSize = Constants.STD_FONT_SIZE.sp, color = Color.Gray
                    )
                }

                Icon(
                    imageVector = Icons.Filled.KeyboardArrowDown,
                    contentDescription = null,
                    tint = Color.Gray,
                    modifier = Modifier.size(24.dp)
                )
            }

            // Dropdown menu
            DropdownMenu(
                expanded = showSceneDropdown,
                onDismissRequest = { showSceneDropdown = false },
                modifier = Modifier.fillMaxWidth(0.8f)
            ) {
                statistics.scenesList.forEachIndexed { index, (sceneName, _) ->
                    DropdownMenuItem(text = { Text(sceneName) }, onClick = {
                        onSceneSelected(index)
                        showSceneDropdown = false
                    })
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Show objects for selected scene
        if (selectedSceneIndex != null && selectedSceneIndex < statistics.objectsBySceneList.size) {
            val sceneEntry = statistics.objectsBySceneList[selectedSceneIndex]
            val objects = sceneEntry.second  // Top 10 objects

            ScrollableItemRow(
                items = objects.map { it.first to it.second },
                sum = objects.sumOf { it.second })
        } else {
            // Placeholder
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(32.dp))
                    .background(Color.White.copy(alpha = 0.2f))
                    .padding(horizontal = 12.dp, vertical = 20.dp),
                contentAlignment = Alignment.Center
            ) {
                Text(
                    text = "Select a scene to view objects",
                    fontSize = Constants.STD_FONT_SIZE.sp,
                    color = Color.White,
                    textAlign = TextAlign.Center
                )
            }
        }
    }
}

@SuppressLint("DefaultLocale")
@Composable
fun StatsPanel(
    avgThreads: Float,
    avgDetectorLatency: Float,
    avgClassifierLatency: Float,
    avgBatteryUsage: Float,
    onClose: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(topStart = 32.dp, topEnd = 32.dp))
            .background(Color.White)  // Pink background
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        Text(
            text = if (AppConfig.mainLanguage.code == "en")
                "More statistics"
            else
                "Mai multe statistici",
            fontSize = 22.sp,
            fontFamily = robotoExtraBold,
            color = Color.Black
        )

        if (avgThreads.toInt() != 0)
        // Stats rows
            StatsRow(
                label = if (AppConfig.mainLanguage.code == "en")
                    "Average number of threads"
                else
                    "Media numărului firelor de executie",
                value = avgThreads.toInt().toString()
            )

        if (avgDetectorLatency.toInt() != 0)
            StatsRow(
                label = "Detector Latency",
                value = "${avgDetectorLatency.toInt()}ms"
            )

        StatsRow(
            label = "Classifier Latency",
            value = "${avgClassifierLatency.toInt()}ms"
        )

        if (avgBatteryUsage.toInt() != 0)
            StatsRow(
                label = if (AppConfig.mainLanguage.code == "en")
                    "Battery Usage Increase"
                else
                    "Creșterea medie a utilizării bateriei",
                value = String.format("%.2f%%", avgBatteryUsage)
            )

        // Header with close button
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = 10.dp)
                .clickable { onClose() },
            horizontalArrangement = Arrangement.Center,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                text = if (AppConfig.mainLanguage.code == "en")
                    "Close"
                else
                    "Închide",
                fontSize = Constants.STD_BUTTON_FONT_SIZE.sp,
                color = colorResource(R.color.error_red),
                fontFamily = robotoExtraBold
            )
        }
    }
}

@Composable
fun StatsRow(label: String, value: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(32.dp))
            .background(Color(0xFFFFC4D6))
            .padding(horizontal = 12.dp, vertical = 12.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = label,
            fontSize = Constants.STD_ERROR_FONT_SIZE.sp,
            color = Color.Black,
            fontFamily = robotoExtraBold
        )

        Text(
            text = value,
            fontSize = Constants.STD_ERROR_FONT_SIZE.sp,
            color = colorResource(R.color.std_purple),
            fontFamily = robotoExtraBold
        )
    }
}

@Preview(
    name = "Environment Reports Activity", showBackground = true, widthDp = 412, heightDp = 917
)
@Composable
fun EnvironmentReportsActivityPreview() {
    val sampleStats = EnvironmentStatistics(
        scenesList = mutableListOf(
            Pair("Kitchen", 145),
            Pair("Bedroom", 98),
            Pair("Living Room", 87),
            Pair("Bathroom", 56),
            Pair("Office", 43),
            Pair("Dining Room", 32),
            Pair("Garage", 21),
            Pair("Garden", 15)
        ),
        objectsList = mutableListOf(
            Pair("Person", 234),
            Pair("Chair", 187),
            Pair("Bottle", 156),
            Pair("Cup", 143),
            Pair("Book", 98),
            Pair("Laptop", 87),
            Pair("Phone", 76),
            Pair("Table", 65),
            Pair("Bed", 54),
            Pair("TV", 43)
        ),
        objectsBySceneList = mutableListOf(
            Pair(
                "Kitchen", mutableListOf(
                    Pair("Bottle", 89),
                    Pair("Cup", 67),
                    Pair("Bowl", 45),
                    Pair("Spoon", 34),
                    Pair("Knife", 28)
                )
            ), Pair(
                "Bedroom", mutableListOf(
                    Pair("Bed", 54),
                    Pair("Book", 43),
                    Pair("Laptop", 32),
                    Pair("Phone", 28),
                    Pair("Chair", 21)
                )
            ), Pair(
                "Living Room", mutableListOf(
                    Pair("TV", 43),
                    Pair("Chair", 76),
                    Pair("Table", 32),
                    Pair("Book", 28),
                    Pair("Person", 98)
                )
            )
        ),
        avgNoThreads = 3.2f,
        avgDetectorLatency = 245.6f,
        avgClassifierLatency = 1834.2f,
        avgPercMoreBatteryUsed = 2.34f
    )

    EnvironmentReportsScreen(
        showLoading = false,
        loadingText = "",
        showErrorDialog = false,
        errorMessage = "",
        showResults = true,
        statistics = sampleStats,
        selectedSceneIndex = null,
        onSceneSelected = {},
        onNavigateHome = {},
        onNavigateReports = {},
        onNavigateSettings = {},
        onErrorRetry = {},
        onErrorOk = {}
    )
}