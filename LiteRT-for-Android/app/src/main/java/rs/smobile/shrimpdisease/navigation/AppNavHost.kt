package rs.smobile.shrimpdisease.navigation

import android.Manifest
import android.content.pm.PackageManager
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.CenterAlignedTopAppBar
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.hilt.lifecycle.viewmodel.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import rs.smobile.shrimpdisease.MainViewModel
import rs.smobile.shrimpdisease.RuntimeDelegate
import rs.smobile.shrimpdisease.ui.history.HistoryScreen
import rs.smobile.shrimpdisease.ui.home.HomeScreen
import rs.smobile.shrimpdisease.ui.inference.InferenceScreen
import rs.smobile.shrimpdisease.ui.settings.SettingsScreen
import rs.smobile.shrimpdisease.utils.BitmapUtils

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AppNavHost(
    modifier: Modifier = Modifier,
    viewModel: MainViewModel = hiltViewModel(),
) {
    val navController = rememberNavController()
    val context = LocalContext.current

    val inputState by viewModel.inputState.collectAsStateWithLifecycle()
    val classificationState by viewModel.classificationState.collectAsStateWithLifecycle()
    val settingsState by viewModel.settingsState.collectAsStateWithLifecycle()
    val modelInfo by viewModel.modelInfo.collectAsStateWithLifecycle()
    val labels by viewModel.labels.collectAsStateWithLifecycle()
    val availableModels by viewModel.availableModels.collectAsStateWithLifecycle()
    val logs by viewModel.logs.collectAsStateWithLifecycle()
    val benchmarkMetrics by viewModel.benchmarkMetrics.collectAsStateWithLifecycle()
    val selectedGroundTruthLabel by viewModel.selectedGroundTruthLabel.collectAsStateWithLifecycle()

    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.CAMERA,
            ) == PackageManager.PERMISSION_GRANTED
        )
    }

    val imagePicker = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri == null) return@rememberLauncherForActivityResult
        runCatching { BitmapUtils.decodeBitmapFromUri(context, uri) }
            .onSuccess { bitmap ->
                viewModel.setGalleryImage(uri, bitmap)
                navController.openInference()
            }
            .onFailure { error ->
                viewModel.showError(error.message ?: "Could not decode selected image.")
            }
    }

    val cameraPermissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        hasCameraPermission = granted
        if (granted) {
            viewModel.startCameraInput()
            navController.openInference()
        } else {
            viewModel.showError("Camera permission is required to take a snapshot.")
        }
    }

    val exportCsvLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.CreateDocument("text/csv")
    ) { uri ->
        if (uri == null) return@rememberLauncherForActivityResult
        val exported = viewModel.exportLogs(context, uri)
        Toast.makeText(
            context,
            if (exported) "Logs exported" else "Export failed",
            Toast.LENGTH_SHORT,
        ).show()
    }

    val currentRoute = navController.currentBackStackEntryAsState().value?.destination?.route
        ?: Screen.Home.route
    val title = screenTitleForRoute(currentRoute)

    Scaffold(
        modifier = modifier,
        topBar = {
            Column {
                CenterAlignedTopAppBar(
                    title = {
                        Text(
                            text = title,
                            style = MaterialTheme.typography.titleLarge,
                            fontWeight = FontWeight.Bold,
                        )
                    },
                    colors = TopAppBarDefaults.topAppBarColors(
                        containerColor = MaterialTheme.colorScheme.surface,
                        titleContentColor = MaterialTheme.colorScheme.onSurface,
                    ),
                )
                if (currentRoute in topLevelScreens.map { it.route }) {
                    PrimaryNavigation(
                        currentRoute = currentRoute,
                        onNavigate = { screen -> navController.navigateTopLevel(screen) },
                    )
                }
            }
        },
    ) { innerPadding ->
        NavHost(
            navController = navController,
            startDestination = Screen.Home.route,
            modifier = Modifier.padding(innerPadding),
        ) {
            composable(Screen.Home.route) {
                HomeScreen(
                    modelInfo = modelInfo,
                    onSelectImage = { imagePicker.launch("image/*") },
                    onOpenCamera = {
                        if (hasCameraPermission) {
                            viewModel.startCameraInput()
                            navController.openInference()
                        } else {
                            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
                        }
                    },
                    onChooseModel = { navController.navigateTopLevel(Screen.Settings) },
                )
            }

            composable(Screen.Inference.route) {
                InferenceScreen(
                    inputState = inputState,
                    classificationState = classificationState,
                    labels = labels,
                    selectedGroundTruthLabel = selectedGroundTruthLabel,
                    modelInfo = modelInfo,
                    benchmarkMetrics = benchmarkMetrics,
                    runtimeDelegateName = settingsState.runtimeDelegate.displayName,
                    debugInfoEnabled = settingsState.showDebugInfo,
                    cameraFps = settingsState.cameraFps,
                    cameraPermissionGranted = hasCameraPermission,
                    onRequestCameraPermission = {
                        cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
                    },
                    onSnapshot = { bitmap ->
                        viewModel.setCameraSnapshot(bitmap)
                    },
                    onFrameObserved = viewModel::recordCameraFrame,
                    onCameraError = viewModel::showError,
                    onGroundTruthSelected = viewModel::setGroundTruthLabel,
                    onSelectAnotherImage = { imagePicker.launch("image/*") },
                    onRunInference = viewModel::runInference,
                    onSaveResult = {
                        val saved = viewModel.saveCurrentResult()
                        Toast.makeText(
                            context,
                            if (saved) "Result logged" else "Run inference before saving",
                            Toast.LENGTH_SHORT,
                        ).show()
                    },
                    onHome = { navController.navigateHomeFromInference() },
                    onHistory = { navController.navigateTopLevel(Screen.History) },
                )
            }

            composable(Screen.History.route) {
                HistoryScreen(
                    logs = logs,
                    benchmarkMetrics = benchmarkMetrics,
                    onExportCsv = { exportCsvLauncher.launch("shrimp_prediction_history.csv") },
                    onClearLogs = viewModel::clearLogs,
                )
            }

            composable(Screen.Settings.route) {
                SettingsScreen(
                    modelInfo = modelInfo,
                    labels = labels,
                    availableModels = availableModels,
                    settingsState = settingsState,
                    benchmarkMetrics = benchmarkMetrics,
                    isLoading = classificationState.isLoading,
                    errorMessage = classificationState.errorMessage,
                    onSelectModel = viewModel::loadModel,
                    onDelegateSelected = { delegate ->
                        if (delegate == RuntimeDelegate.CPU) {
                            viewModel.setRuntimeDelegate(delegate)
                        }
                    },
                    onThresholdChange = viewModel::setConfidenceThreshold,
                    onShowDebugInfoChange = viewModel::setShowDebugInfo,
                    onResetMetrics = viewModel::clearLogs,
                )
            }
        }
    }
}

@Composable
private fun PrimaryNavigation(
    currentRoute: String,
    onNavigate: (Screen) -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 8.dp),
    ) {
        topLevelScreens.forEach { screen ->
            val selected = currentRoute == screen.route
            if (selected) {
                Button(
                    onClick = { onNavigate(screen) },
                    modifier = Modifier
                        .weight(1f)
                        .padding(horizontal = 4.dp),
                ) {
                    Text(text = screen.title)
                }
            } else {
                TextButton(
                    onClick = { onNavigate(screen) },
                    modifier = Modifier
                        .weight(1f)
                        .padding(horizontal = 4.dp),
                ) {
                    Text(text = screen.title)
                }
            }
        }
    }
}

private fun NavHostController.navigateTopLevel(screen: Screen) {
    navigate(screen.route) {
        popUpTo(graph.findStartDestination().id) {
            saveState = true
        }
        launchSingleTop = true
        restoreState = true
    }
}

private fun NavHostController.openInference() {
    navigate(Screen.Inference.route) {
        launchSingleTop = true
    }
}

private fun NavHostController.navigateHomeFromInference() {
    navigate(Screen.Home.route) {
        popUpTo(Screen.Home.route) {
            inclusive = false
        }
        launchSingleTop = true
    }
}

private fun screenTitleForRoute(route: String): String {
    return when (route) {
        Screen.Home.route -> "Shrimp Disease"
        Screen.Inference.route -> Screen.Inference.title
        Screen.History.route -> Screen.History.title
        Screen.Settings.route -> Screen.Settings.title
        else -> "Shrimp Disease"
    }
}
