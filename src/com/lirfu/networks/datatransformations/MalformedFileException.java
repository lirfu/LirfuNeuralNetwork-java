package com.lirfu.networks.datatransformations;

import java.io.IOException;

/**
 * Created by lirfu on 05.09.17..
 */
public class MalformedFileException extends IOException {
    public MalformedFileException(){
        super();
    }
    public MalformedFileException(String message){
        super(message);
    }
}
