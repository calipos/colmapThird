#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
import OpenGL.GL as gl
import imgui
import sys


backend = "glfw"
if "sdl2" in sys.argv:
    backend = "sdl2"
elif "pygame" in sys.argv:
    backend = "pygame"
elif "glfw" in sys.argv:
    backend = "glfw"
elif "cocos2d" in sys.argv:
    backend = "cocos2d"
sys.stderr.write("%s backend selected\n" % backend)

if backend == "sdl2":
    from sdl2 import *
    import ctypes
    from imgui.integrations.sdl2 import SDL2Renderer
elif backend == "pygame":
    import pygame
    from imgui.integrations.pygame import PygameRenderer
elif backend == "glfw":
    import glfw
    from imgui.integrations.glfw import GlfwRenderer
elif backend == "cocos2d":
    import cocos
    from cocos.director import director
    from imgui.integrations.cocos2d import ImguiLayer


def main_glfw():
    def glfw_init():
        width, height = 1280, 720
        window_name = "minimal ImGui/GLFW3 example"
        if not glfw.init():
            print("Could not initialize OpenGL context")
            sys.exit(1)
        # OS X supports only forward-compatible core profiles from 3.2
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)
        # Create a windowed mode window and its OpenGL context
        window = glfw.create_window(
            int(width), int(height), window_name, None, None)
        glfw.make_context_current(window)
        if not window:
            glfw.terminate()
            print("Could not initialize Window")
            sys.exit(1)
        return window

    window = glfw_init()
    impl = GlfwRenderer(window)
    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()
        imgui.new_frame()
        on_frame()
        gl.glClearColor(0.0, 0.0, 0.0, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)
    impl.shutdown()
    glfw.terminate()



# backend-independent frame rendering function:
def on_frame():
    if imgui.begin_main_menu_bar():
        if imgui.begin_menu("File", True):
            clicked_quit, selected_quit = imgui.menu_item(
                "Quit", "Cmd+Q", False, True)
            if clicked_quit:
                sys.exit(0)
            imgui.end_menu()
        imgui.end_main_menu_bar()
    imgui.show_test_window()
    imgui.begin("Custom window", True)
    imgui.text("Bar")
    imgui.text_colored("Eggs", 0.2, 1.0, 0.0)
    imgui.end()


if __name__ == "__main__":
    imgui.create_context()
    io = imgui.get_io()
    main_glfw()
