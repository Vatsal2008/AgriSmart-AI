package com.agrismart.app

import com.agrismart.app.R
import android.os.Bundle
import androidx.activity.compose.setContent
import androidx.appcompat.app.AppCompatActivity
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Menu
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.DrawerValue
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalDrawerSheet
import androidx.compose.material3.ModalNavigationDrawer
import androidx.compose.material3.NavigationDrawerItem
import androidx.compose.material3.NavigationDrawerItemDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.material3.rememberDrawerState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import com.agrismart.app.core.designsystem.components.MockBanner
import com.agrismart.app.core.designsystem.theme.AgriSmartTheme
import com.agrismart.app.core.model.ModelDownloader
import com.agrismart.app.data.api.ApiConfig
import com.agrismart.app.data.repo.CropRepository
import com.agrismart.app.data.repo.PredictionRepository
import com.agrismart.app.domain.model.PredictionResult
import com.agrismart.app.feature.help.HelpAboutScreen
import com.agrismart.app.feature.history.RecentTestsScreen
import com.agrismart.app.feature.home.HomeScreen
import com.agrismart.app.feature.onboarding.LanguageScreen
import com.agrismart.app.feature.onboarding.ModelDownloadScreen
import com.agrismart.app.feature.scan.AddPhotoScreen
import com.agrismart.app.feature.scan.CheckingScreen
import com.agrismart.app.feature.scan.ChoosePlantScreen
import com.agrismart.app.feature.scan.LiveCameraScreen
import com.agrismart.app.feature.scan.ResultScreen
import com.agrismart.app.feature.settings.SettingsScreen
import dagger.hilt.android.AndroidEntryPoint
import kotlinx.coroutines.launch
import javax.inject.Inject

@AndroidEntryPoint
class MainActivity : AppCompatActivity() {

    @Inject
    lateinit var apiConfig: ApiConfig

    @Inject
    lateinit var modelDownloader: ModelDownloader

    @Inject
    lateinit var cropRepository: CropRepository

    @Inject
    lateinit var predictionRepository: PredictionRepository

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            AgriSmartTheme {
                AgriSmartAppContent(
                    apiConfig = apiConfig,
                    modelDownloader = modelDownloader,
                    cropRepository = cropRepository,
                    predictionRepository = predictionRepository
                )
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AgriSmartAppContent(
    apiConfig: ApiConfig,
    modelDownloader: ModelDownloader,
    cropRepository: CropRepository,
    predictionRepository: PredictionRepository
) {
    val navController = rememberNavController()
    val drawerState = rememberDrawerState(initialValue = DrawerValue.Closed)
    val scope = rememberCoroutineScope()

    val navBackStackEntry = navController.currentBackStackEntryAsState()
    val currentRoute = navBackStackEntry.value?.destination?.route

    val isFullscreenRoute = currentRoute == "language" || currentRoute == "model_download"

    var currentResult by remember { mutableStateOf<PredictionResult?>(null) }

    ModalNavigationDrawer(
        drawerState = drawerState,
        gesturesEnabled = !isFullscreenRoute,
        drawerContent = {
            ModalDrawerSheet(
                modifier = Modifier.width(300.dp),
                drawerContainerColor = MaterialTheme.colorScheme.surface
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(16.dp)
                ) {
                    Text(
                        text = stringResource(R.string.app_name),
                        style = MaterialTheme.typography.headlineLarge,
                        color = MaterialTheme.colorScheme.primary,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.padding(vertical = 16.dp)
                    )

                    Spacer(modifier = Modifier.height(16.dp))

                    NavigationDrawerItem(
                        icon = { Icon(Icons.Default.Home, contentDescription = null) },
                        label = { Text(stringResource(R.string.nav_check_plant)) },
                        selected = currentRoute == "home" || currentRoute == "choose_plant",
                        onClick = {
                            scope.launch { drawerState.close() }
                            navController.navigate("home") {
                                popUpTo("home") { inclusive = true }
                            }
                        },
                        modifier = Modifier.padding(NavigationDrawerItemDefaults.ItemPadding)
                    )

                    NavigationDrawerItem(
                        icon = { Icon(painterResource(id = R.drawable.ic_photo_camera), contentDescription = null) },
                        label = { Text("Live Camera Scan") },
                        selected = currentRoute == "live_camera",
                        onClick = {
                            scope.launch { drawerState.close() }
                            navController.navigate("live_camera")
                        },
                        modifier = Modifier.padding(NavigationDrawerItemDefaults.ItemPadding)
                    )

                    NavigationDrawerItem(
                        icon = { Icon(Icons.Default.Info, contentDescription = null) },
                        label = { Text(stringResource(R.string.nav_recent_tests)) },
                        selected = currentRoute == "history",
                        onClick = {
                            scope.launch { drawerState.close() }
                            navController.navigate("history")
                        },
                        modifier = Modifier.padding(NavigationDrawerItemDefaults.ItemPadding)
                    )

                    NavigationDrawerItem(
                        icon = { Icon(Icons.Default.Settings, contentDescription = null) },
                        label = { Text(stringResource(R.string.nav_settings)) },
                        selected = currentRoute == "settings",
                        onClick = {
                            scope.launch { drawerState.close() }
                            navController.navigate("settings")
                        },
                        modifier = Modifier.padding(NavigationDrawerItemDefaults.ItemPadding)
                    )

                    NavigationDrawerItem(
                        icon = { Icon(Icons.Default.Info, contentDescription = null) },
                        label = { Text(stringResource(R.string.nav_help_about)) },
                        selected = currentRoute == "help",
                        onClick = {
                            scope.launch { drawerState.close() }
                            navController.navigate("help")
                        },
                        modifier = Modifier.padding(NavigationDrawerItemDefaults.ItemPadding)
                    )

                    Spacer(modifier = Modifier.weight(1f))

                    Text(
                        text = "Version 1.0.0 (model-v2)",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
                        modifier = Modifier.padding(16.dp)
                    )
                }
            }
        }
    ) {
        Scaffold(
            topBar = {
                if (!isFullscreenRoute) {
                    Column {
                        TopAppBar(
                            title = {
                                Text(
                                    text = stringResource(R.string.app_name),
                                    style = MaterialTheme.typography.titleLarge,
                                    fontWeight = FontWeight.Bold
                                )
                            },
                            navigationIcon = {
                                IconButton(onClick = { scope.launch { drawerState.open() } }) {
                                    Icon(
                                        imageVector = Icons.Default.Menu,
                                        contentDescription = stringResource(R.string.tour_menu)
                                    )
                                }
                            },
                            colors = TopAppBarDefaults.topAppBarColors(
                                containerColor = MaterialTheme.colorScheme.surface,
                                titleContentColor = MaterialTheme.colorScheme.onSurface
                            )
                        )
                        if (!apiConfig.isConfigured) {
                            MockBanner()
                        }
                    }
                }
            }
        ) { innerPadding ->
            NavHost(
                navController = navController,
                startDestination = "language",
                modifier = Modifier.padding(innerPadding)
            ) {
                composable("language") {
                    LanguageScreen(
                        onLanguageSelected = { tag ->
                            navController.navigate("model_download")
                        }
                    )
                }

                composable("model_download") {
                    ModelDownloadScreen(
                        modelDownloader = modelDownloader,
                        onDownloadComplete = {
                            navController.navigate("home") {
                                popUpTo("language") { inclusive = true }
                            }
                        }
                    )
                }

                composable("home") {
                    HomeScreen(
                        onCheckPlantClick = { navController.navigate("choose_plant") },
                        onLiveCameraClick = { navController.navigate("live_camera") },
                        onRecentTestsClick = { navController.navigate("history") }
                    )
                }

                composable("live_camera") {
                    LiveCameraScreen(
                        predictionRepository = predictionRepository,
                        onResultConfirmed = { result ->
                            currentResult = result
                            navController.navigate("result")
                        }
                    )
                }

                composable("choose_plant") {
                    ChoosePlantScreen(
                        cropRepository = cropRepository,
                        onCropSelected = { cropId ->
                            navController.navigate("add_photo/$cropId")
                        }
                    )
                }

                composable(
                    route = "add_photo/{cropId}",
                    arguments = listOf(navArgument("cropId") { type = NavType.StringType })
                ) { backStack ->
                    val cropId = backStack.arguments?.getString("cropId") ?: "tomato"
                    AddPhotoScreen(
                        cropId = cropId,
                        onTakePhotoClick = { navController.navigate("checking/$cropId") },
                        onChooseFromPhoneClick = { navController.navigate("checking/$cropId") }
                    )
                }

                composable(
                    route = "checking/{cropId}",
                    arguments = listOf(navArgument("cropId") { type = NavType.StringType })
                ) { backStack ->
                    val cropId = backStack.arguments?.getString("cropId") ?: "tomato"
                    CheckingScreen(
                        cropId = cropId,
                        predictionRepository = predictionRepository,
                        onPredictionComplete = { result ->
                            currentResult = result
                            navController.navigate("result") {
                                popUpTo("choose_plant") { inclusive = true }
                            }
                        }
                    )
                }

                composable("result") {
                    currentResult?.let { res ->
                        ResultScreen(
                            result = res,
                            onCheckAnotherClick = { navController.navigate("choose_plant") }
                        )
                    }
                }

                composable("history") {
                    RecentTestsScreen(
                        predictionRepository = predictionRepository,
                        onTestClick = { }
                    )
                }

                composable("settings") {
                    SettingsScreen(
                        onLanguageClick = { navController.navigate("language") },
                        apiConfig = apiConfig
                    )
                }

                composable("help") {
                    HelpAboutScreen()
                }
            }
        }
    }
}
