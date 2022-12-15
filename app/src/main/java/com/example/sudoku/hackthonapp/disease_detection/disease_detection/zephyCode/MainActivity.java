package com.example.sudoku.hackthonapp.disease_detection.disease_detection.zephyCode;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.example.sudoku.hackthonapp.disease_detection.disease_detection.zephyCode.ml.Plantsdiseases;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Objects;

public class MainActivity extends AppCompatActivity {

    TextView demoText, classified, clickHere, result;
    ImageView imageView;
    Button picture;
    int REQUEST_CAMERA_CAPTURE = 1;

    int imageSize = 224; // default image size

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Objects.requireNonNull(getSupportActionBar()).hide();
        result = findViewById(R.id.result);
        imageView = findViewById(R.id.imageView);
        picture = findViewById(R.id.picture);
        demoText = findViewById(R.id.demoText);
        clickHere = findViewById(R.id.click_here);
        classified = findViewById(R.id.classified);

        demoText.setVisibility(View.VISIBLE);
        clickHere.setVisibility(View.GONE);
        classified.setVisibility(View.GONE);
        result.setVisibility(View.GONE);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (requestCode == 101) {
            super.onActivityResult(requestCode, resultCode, data);
            assert data != null;
            Bitmap image = (Bitmap) data.getExtras().get("data");
            int dimension = Math.min(image.getWidth(), image.getHeight());
            image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
            imageView.setImageBitmap(image);

            demoText.setVisibility(View.GONE);
            clickHere.setVisibility(View.VISIBLE);
            classified.setVisibility(View.VISIBLE);
            result.setVisibility(View.VISIBLE);

            image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
            classifyImage(image);
        }
    }

    private void classifyImage(Bitmap image) {
        try {
            Plantsdiseases model = Plantsdiseases.newInstance(getApplicationContext());
            TensorBuffer inputFeatures = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            int[] intValue = new int[imageSize * imageSize];
            image.getPixels(intValue, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
            int pixel = 0;
            for (int i = 0; i < imageSize; i++) {
                for (int j = 0; j < imageSize; j++) {
                    int val = intValue[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF)*(1.f / 255.f));
                    byteBuffer.putFloat(((val >> 8) & 0xFF)*(1.f / 255.f));
                    byteBuffer.putFloat(((val & 16) & 0xFF)*(1.f / 255.f));
                }
            }

            inputFeatures.loadBuffer(byteBuffer);
            // Runs model interface and gets results for us.
            Plantsdiseases.Outputs outputs = model.process(inputFeatures);
            TensorBuffer outputFeatures = outputs.getOutputFeature0AsTensorBuffer();
            float[] confidence = outputFeatures.getFloatArray();

            // Find the index of the class with the biggest confidence
            int maxPos = 0;
            float maxConfidence = 0;
            for (int i = 0; i < confidence.length; i++) {
                if (confidence[i] > maxConfidence) {
                    maxConfidence = confidence[i];
                    maxPos = i;
                }
            }
            String[] classes = {"Pepper Bell Bacterial Spot", "Pepper Bell Healthy", "Plant Early Blight", "Potato Healthy", "Potato Late Blight", "Tomato Bacterial Spot", "Tomato Early Blight", "Tomato Healthy", "Tomato Late Blight", "Tomato Leaf Mold"};
            result.setText(classes[maxPos]);
            result.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View view) {
                    // To search the disease on Web Browser
                    startActivity(new Intent(Intent.ACTION_VIEW, Uri.parse("https://www.google.com/search?q="+result.getText())));
                }
            });
            model.close();
        } catch (IOException e) {
            // TODO HANDLE THE EXCEPTION
        }
    }

    public void setPictureButton(View view) {
        if (ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            // Launch Camera if the permission is granted.
            Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            startActivityForResult(cameraIntent, REQUEST_CAMERA_CAPTURE);
        } else {
            // Request for permission if it is not granted.
            ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.CAMERA}, 101);
        }
        Toast.makeText(MainActivity.this, "I was clicked", Toast.LENGTH_SHORT).show();
    }
}