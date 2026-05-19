package rs.smobile.shrimpdisease.navigation

sealed class Screen(
    val route: String,
    val title: String,
) {
    object Home : Screen("home", "Home")
    object Inference : Screen("inference", "Inference")
    object History : Screen("history", "History")
    object Settings : Screen("settings", "Profile")
}

val topLevelScreens = listOf(
    Screen.Home,
    Screen.History,
    Screen.Settings,
)
