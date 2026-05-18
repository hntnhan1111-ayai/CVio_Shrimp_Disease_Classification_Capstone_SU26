package rs.smobile.shrimpdisease

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import dagger.hilt.android.AndroidEntryPoint
import rs.smobile.shrimpdisease.navigation.AppNavHost
import rs.smobile.shrimpdisease.ui.theme.ShrimpDiseaseTheme

@AndroidEntryPoint
class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        enableEdgeToEdge()
        setContent {
            ShrimpDiseaseTheme {
                AppNavHost()
            }
        }
    }
}
