plugins {
  alias(libs.plugins.android.application)
  alias(libs.plugins.compose.compiler)
  alias(libs.plugins.kotlin.serialization)
}

android {
  namespace = "ru.hexteam.stenograf"
  compileSdk = 36

  defaultConfig {
    applicationId = "ru.hexteam.stenograf"
    minSdk = 26
    targetSdk = 36
    versionCode = 1
    versionName = "1.0.0"

    ndk {
      // arm64-v8a — реальные смартфоны, x86_64 — эмулятор Android Studio
      abiFilters += listOf("arm64-v8a", "x86_64")
    }
  }

  buildTypes {
    release {
      // JNI (sherpa-onnx) и рефлексия (LiteRT-LM) не дружат с обфускацией
      isMinifyEnabled = false
      isShrinkResources = false
      // Подписываем debug-ключом, чтобы эксперты могли сразу установить APK
      signingConfig = signingConfigs.getByName("debug")
      proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
    }
  }

  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
  }

  buildFeatures {
    compose = true
    buildConfig = true
    aidl = false
    shaders = false
  }

  androidResources {
    // Модели читаются напрямую из APK — не сжимаем их
    noCompress += listOf("onnx", "litertlm", "txt")
  }

  packaging {
    resources {
      excludes += "/META-INF/{AL2.0,LGPL2.1}"
    }
    jniLibs {
      useLegacyPackaging = false
    }
  }
}

kotlin {
  jvmToolchain(17)
}

dependencies {
  val composeBom = platform(libs.androidx.compose.bom)
  implementation(composeBom)

  implementation(libs.androidx.core.ktx)
  implementation(libs.androidx.lifecycle.runtime.ktx)
  implementation(libs.androidx.activity.compose)
  implementation(libs.androidx.lifecycle.runtime.compose)
  implementation(libs.androidx.lifecycle.viewmodel.compose)

  implementation(libs.androidx.compose.ui)
  implementation(libs.androidx.compose.ui.tooling.preview)
  implementation(libs.androidx.compose.material3)
  implementation(libs.androidx.compose.material.icons)
  debugImplementation(libs.androidx.compose.ui.tooling)

  implementation(libs.kotlinx.coroutines.android)
  implementation(libs.kotlinx.serialization.json)

  // Распознавание речи, VAD, диаризация, шумоподавление (ONNX Runtime)
  implementation(files("libs/sherpa-onnx-2.0.1.aar"))
  // Локальная LLM (Qwen3-0.6B) для протокола встречи
  implementation(libs.litertlm.android)

  testImplementation(libs.junit)
}
