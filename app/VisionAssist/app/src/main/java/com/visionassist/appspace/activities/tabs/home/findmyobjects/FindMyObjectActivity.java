package com.visionassist.appspace.activities.tabs.home.findmyobjects;

import android.os.Bundle;
import android.util.Log;
import androidx.appcompat.app.AppCompatActivity;
import com.visionassist.appspace.R;
import com.visionassist.appspace.utils.Constants;

import java.util.ArrayList;

public class FindMyObjectActivity extends AppCompatActivity {

    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_main);

        ArrayList<Integer> matchedIndices = getIntent().getIntegerArrayListExtra(Constants.EXTRA_MATCHED_INDICES);
        StringBuilder indicesBuilder = new StringBuilder();
        indicesBuilder.append("Matched Indices Array:\n");
        indicesBuilder.append("Values:\n[");
        for (int i = 0; i < matchedIndices.size(); i++) {
            indicesBuilder.append(matchedIndices.get(i));
            if (i < matchedIndices.size() - 1) {
                indicesBuilder.append(", ");
            }
        }
        indicesBuilder.append("]");
        Log.i("FindMyObjectActivity",indicesBuilder.toString());
    }
}