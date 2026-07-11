package com.visionassist.appspace;

import com.visionassist.appspace.jetpack.managers.LoadingManager;

public class ExceptionVisionAssist extends Exception {
    private final int errorCode;
    private final LoadingManager loadingManager;

    public ExceptionVisionAssist(int errorCode, LoadingManager loadingManager) {
        super("nothing to declare");
        this.errorCode = errorCode;
        this.loadingManager = loadingManager;
    }

    public int getErrorCode() {
        return errorCode;
    }

    public LoadingManager getLoadingManager() {
        return loadingManager;
    }
}