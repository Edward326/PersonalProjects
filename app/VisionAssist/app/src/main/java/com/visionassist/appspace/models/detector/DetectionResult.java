package com.visionassist.appspace.models.detector;
import android.graphics.RectF;
import java.util.List;

public class DetectionResult {
    private List<RectF> boundingBoxes;
    private List<Float> confidences;
    private List<String> labels;

    public DetectionResult(List<RectF> boundingBoxes, List<Float> confidences, List<String> labels) {
        this.boundingBoxes = boundingBoxes;
        this.confidences = confidences;
        this.labels = labels;
    }

    public List<RectF> getBoundingBoxes() {
        return boundingBoxes;
    }

    public void setBoundingBoxes(List<RectF> boundingBoxes) {
        this.boundingBoxes = boundingBoxes;
    }

    public List<Float> getConfidences() {
        return confidences;
    }

    public void setConfidences(List<Float> confidences) {
        this.confidences = confidences;
    }

    public List<String> getLabels() {
        return labels;
    }

    public void setLabels(List<String> labels) {
        this.labels = labels;
    }

    public int getDetectionCount() {
        return labels != null ? labels.size() : 0;
    }

    public boolean hasDetections() {
        return getDetectionCount() > 0;
    }

    public String listBoundingBoxes(){
        String resultConcat = "";
        for(int i=0;i< labels.size();i++)
        {
            resultConcat+=String.format("%d. %s(%.3f)",i+1,labels.get(i),confidences.get(i));
            if(i+1< labels.size())
                resultConcat+="\n";
        }
        return resultConcat;
    }
}