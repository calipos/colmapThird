#include "imgui_tools.h"

bool LoadTextureFromMat(const cv::Mat& originalImg, GLuint* out_texture, int* out_width, int* out_height)
{
    cv::Mat img;
    if (originalImg.channels() == 1)
    {
        cv::cvtColor(originalImg, img, cv::COLOR_GRAY2RGBA);
    }
    else
    {
        cv::cvtColor(originalImg, img, cv::COLOR_BGR2RGBA);
    }
    unsigned char* image_data = img.data;
    if (image_data == NULL)
        return false;
    int image_width = img.cols;
    int image_height = img.rows;
    // Create a OpenGL texture identifier
    GLuint image_texture;    glGenTextures(1, &image_texture);    glBindTexture(GL_TEXTURE_2D, image_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
#if defined(GL_UNPACK_ROW_LENGTH) && !defined(__EMSCRIPTEN__)
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
#endif
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width, image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data);

    *out_texture = image_texture;
    *out_width = image_width;
    *out_height = image_height;

    return true;
}

