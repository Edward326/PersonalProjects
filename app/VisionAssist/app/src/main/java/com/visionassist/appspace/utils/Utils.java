package com.visionassist.appspace.utils;

import android.content.Context;
import java.io.IOException;
import java.io.InputStream;

public class Utils {

    public static int checkProfile(Context context) {
        if(!FileUtils.assetExists(context, Constants.PROFILE_FILE))
            return 1;
        try {
            InputStream inputStream = context.getAssets().open(Constants.PROFILE_FILE);
            return JSONValidation.validateProfile(inputStream,false);
        }catch (IOException e) {
            return 1;
        }
    }
}