package com.visionassist.appspace.activities.tabs.reports

data class EnvironmentStatistics(
    // 1. Most common scenes: sorted by occurrence count (descending)
    val scenesList: MutableList<Pair<String, Int>> = mutableListOf(),

    // 2. Most common objects: sorted by occurrence count (descending)
    val objectsList: MutableList<Pair<String, Int>> = mutableListOf(),

    // 3. Objects by scene: for each scene, list of objects sorted by count
    val objectsBySceneList: MutableList<Pair<String, MutableList<Pair<String, Int>>>> = mutableListOf(),

    // 4. Average number of threads used
    var avgNoThreads: Float = 0f,

    // 5. Average detector latency (ms)
    var avgDetectorLatency: Float = 0f,

    // 6. Average classifier latency (ms)
    var avgClassifierLatency: Float = 0f,

    // 7. Average battery usage increase (%)
    var avgPercMoreBatteryUsed: Float = 0f,

    // Internal counters for averaging
    @Transient var detectionRecordCount: Int = 0,
    @Transient var captionRecordCount: Int = 0
) {
    fun sortAllByOccurrence() {
        // Sort scenes by count
        scenesList.sortByDescending { it.second }

        // Sort objects by count
        objectsList.sortByDescending { it.second }

        // Sort objects within each scene by count
        objectsBySceneList.forEach { (_, objects) ->
            objects.sortByDescending { it.second }
        }
    }
}