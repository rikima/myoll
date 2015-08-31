package com.rikima.utils;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.LineNumberReader;

public class LineIterator {
    private LineNumberReader lnr;
    private String line = null;
    
    protected LineIterator(File file) throws FileNotFoundException {
        lnr = new LineNumberReader(new InputStreamReader(new FileInputStream(file)));
    }

    public boolean hasNext() throws IOException {
        this.line = this.lnr.readLine();
        return this.line != null;
    }

    public String next() throws IOException {
        return this.line;
    }
}
