package com.contextmesh;

public class MessagePayload {
    public String type;
    public String content;

    MessagePayload(String type, String content) {
        this.type = type;
        this.content = content;
    }
}
