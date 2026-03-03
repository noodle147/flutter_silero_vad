package com.example.flutter_silero_vad

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.nio.FloatBuffer

class VadIteratorV6 constructor(
    modelPath: String,
    // model config
    private val sampleRate: Long,
    frameSize: Long,
    private val threshold: Float,
    minSilenceDurationMs: Long,
    speechPadMs: Long
) : VadIterator {

    private lateinit var model: SlieroVadOnnxModelV6;

     init {
        this.model = SlieroVadOnnxModelV6(modelPath);
    }

    override fun resetState() {
       model.resetStates();
    }

    override fun predict(data: FloatArray): Boolean {
        val result = model.call(arrayOf(data), sampleRate.toInt())
        println("VAD prediction result: ${result.size} ${result[0]}, threshold: $threshold")
        return result[0] > threshold
    }

    override fun release() {
        model.close();
    }
}