package com.rikima.utils;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class IoUtil {
    public static LineIterator iterator(File file) throws FileNotFoundException {
        return new LineIterator(file);
    }

    public static void output(String path, Object o) throws IOException {
        ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(path));
        out.writeObject(o);
        out.close();
    }

    public static void outputAsText(String path, String data) throws IOException {
        FileWriter fw = new FileWriter(new File(path));
        fw.write(data);
        fw.close();
    }

    public static Object load(String path) throws IOException, ClassNotFoundException {
        ObjectInputStream in = new ObjectInputStream(new FileInputStream(path));
        Object o = in.readObject();
        in.close();
        return o;
    }

    public static String read(File file) throws IOException {
        String text = "";
        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file)));
        String l = null;
        while ((l = br.readLine()) != null) {
            text += l + System.getProperty("line.separator");
        }
        return text;
    }

    public static List<String> readAsList(File file) throws IOException {
        List<String> res = new ArrayList<String>();
        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file)));
        String l = null;
        while ((l = br.readLine()) != null) {
            res.add(l.trim());
        }
        return res;
    }
}
