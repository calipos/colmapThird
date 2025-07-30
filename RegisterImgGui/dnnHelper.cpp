#include "dnnHelper.h"
namespace dnn
{
    namespace ncnnHelper
    {
        cv::dnn::MatShape getBlobShape(const ncnn::Mat& out)
        {
            int dim = out.dims;
            cv::dnn::MatShape ret;
            switch (dim)
            {
            case 4:
                ret = cv::dnn::MatShape{ out.c ,out.d ,out.h ,out.w };
                break;
            case 3:
                ret = cv::dnn::MatShape{ out.c ,out.h ,out.w };
                break;
            case 2:
                ret = cv::dnn::MatShape{ out.h ,out.w };
                break;
            case 1:
                ret = cv::dnn::MatShape{ out.w };
                break;
            default:
                break;
            }
            return ret;
        }
        void convertImgToMemFile(const std::string& path)
        {
            cv::Mat img = cv::imread(path);
            cv::Mat ret;
            if (!img.empty())
            {
                std::fstream fout(path + ".dat", std::ios::out | std::ios::binary);
                int height = img.rows;
                int width = img.cols;
                ret = cv::Mat(height, width, CV_8UC3);
                fout.write((const char*)&height, sizeof(int));
                fout.write((const char*)&width, sizeof(int));
                std::vector<uchar>dat(height * width * 3);
                for (int r = 0; r < height; r++)
                {
                    for (int c = 0; c < width; c++)
                    {
                        uchar d1 = img.at<cv::Vec3b>(r, c)[0];
                        uchar d2 = img.at<cv::Vec3b>(r, c)[1];
                        uchar d3 = img.at<cv::Vec3b>(r, c)[2];
                        int idx = r * width + c;
                        dat[3 * idx] = d1;
                        dat[3 * idx + 1] = d2;
                        dat[3 * idx + 2] = d3;
                        ret.at<cv::Vec3b>(r, c)[0] = d1;
                        ret.at<cv::Vec3b>(r, c)[1] = d2;
                        ret.at<cv::Vec3b>(r, c)[2] = d3;
                    }
                }
                fout.write((char*)&dat[0], dat.size() * sizeof(char));
                fout.close();
            }
            return;
        }
        cv::Mat recoverFromMemfile(const std::string& path)
        {
            cv::Mat img = cv::imread("../a.bmp");
            cv::Mat ret;
            int height = 0;
            int width = 0;
            int dataLength = 0;
            std::fstream fin(path, std::ios::in | std::ios::binary);
            fin.read((char*)&height, sizeof(int));
            fin.read((char*)&width, sizeof(int));
            if (height > 0 && width > 0)
            {
                std::vector<uchar>dat(height * width * 3);
                fin.read((char*)&dat[0], dat.size() * sizeof(char));
                ret = cv::Mat(height, width, CV_8UC3);
                for (int r = 0; r < height; r++)
                {
                    for (int c = 0; c < width; c++)
                    {
                        int idx = r * width + c;
                        uchar d1 = dat[3 * idx];
                        uchar d2 = dat[3 * idx + 1];
                        uchar d3 = dat[3 * idx + 2];
                        ret.at<cv::Vec3b>(r, c)[0] = d1;
                        ret.at<cv::Vec3b>(r, c)[1] = d2;
                        ret.at<cv::Vec3b>(r, c)[2] = d3;


                    }
                }
            }
            return ret;
        }
        std::vector<int>getDenominators(const cv::dnn::MatShape& shape)
        {
            std::vector<int>denominators(shape.size(), 1);
            for (int i = shape.size() - 2; i >= 0; i--)
            {
                denominators[i] = denominators[i + 1] * shape[i + 1];
            }
            denominators.back() = 1;
            return denominators;
        }
        std::vector<int>getPos(const int& idx, const std::vector<int>& denominators)
        {
            std::vector<int>pos(denominators.size(), 0);
            pos[0] = idx;
            for (size_t i = 0; i < denominators.size(); i++)
            {
                if (i + 1 < denominators.size())
                {
                    pos[i + 1] = pos[i] % denominators[i];
                }
                pos[i] /= denominators[i];
            }
            return pos;
        }
        std::ostream& printBlob(const ncnn::Mat& out,std::ostream&os)
        {
            ncnn::Mat shape = out.shape();
            os << "out shape = " << shape.c << " " << shape.d << " " << shape.h << " " << shape.w << ")  dim=" << out.dims << std::endl;
            int cstep = out.cstep;
            int dstep = out.cstep;
            const float* data = (float*)out.data;
            const int elemSize = out.elemsize;
            if (shape.d > 1) dstep /= shape.d;
            for (int c = 0; c < shape.c; c++)
            {
                for (int d = 0; d < shape.d; d++)
                {
                    data = (float*)out.data + d * dstep + c * cstep;
                    for (int h = 0; h < shape.h; ++h)
                    {
                        const float* data2 = data + h * shape.w;
                        os << "[" << c << "," << d << "," << h << ":];  ";
                        for (int w = 0; w < shape.w; ++w)
                        {
                            os << data2[w] << " ";
                            if (w == 2)
                            {
                                int newW = shape.w - 4;
                                if (newW > w)
                                {
                                    os << " ... ";
                                    w = newW;
                                }
                            }
                        }
                        os << std::endl;
                        if (h == 2)
                        {
                            int newH = shape.h - 4;
                            if (newH > h)
                            {
                                os << " ... " << std::endl;
                                h = newH;
                            }
                        }
                    }
                    if (d == 2)
                    {
                        int newD = shape.d - 4;
                        if (newD > d)
                        {
                            os << " ... " << std::endl;
                            d = newD;
                        }
                    }
                    else
                    {
                        os << std::endl;
                    }
                }
                if (c == 2)
                {
                    int newC = shape.c - 4;
                    if (newC > c)
                    {
                        os << " ... " << std::endl;
                        c = newC;
                    }
                }
            }
            return os;
        }
        void writeBlob(const std::string& path, const ncnn::Mat& out)
        {
            std::fstream fout(path, std::ios::out | std::ios::binary);
            cv::dnn::MatShape shape = getBlobShape(out);
            int dims = shape.size();
            fout.write((char*)&dims, sizeof(int));
            fout.write((char*)&shape[0], dims * sizeof(int));
            std::vector<int>denominators = getDenominators(shape);
            int cstep = out.cstep;
            int dstep = 0;
            if (shape.size() == 4)
            {
                dstep = shape[2] * shape[3];
            }
            int total = 1;
            for (int c = 0; c < shape.size(); c++)
            {
                total *= shape[c];
            }
            for (int i = 0; i < total; i++)
            {
                std::vector<int>pos = getPos(i, denominators);
                {
                    int c = 0;
                    int d = 0;
                    int h = 0;
                    int w = 0;
                    if (pos.size() == 4)
                    {
                        c = pos[0];
                        d = pos[1];
                        h = pos[2];
                        w = pos[3];
                    }
                    if (pos.size() == 3)
                    {
                        c = pos[0];
                        h = pos[1];
                        w = pos[2];
                    }
                    if (pos.size() == 2)
                    {
                        h = pos[0];
                        w = pos[1];
                    }
                    if (pos.size() == 1)
                    {
                        w = pos[0];
                    }
                    float* data = (float*)out.data + d * dstep + c * cstep;
                    const float* data2 = data + h * out.w + w;
                    fout.write((char*)data2, sizeof(float));
                }
            }
            fout.close();
            return;
        }
        bool serializationBlob(const ncnn::Mat& out, cv::dnn::MatShape& shape, std::vector<float>& dat)
        {
            shape = getBlobShape(out);
            int dims = shape.size();
            std::vector<int>denominators = getDenominators(shape);
            int cstep = out.cstep;
            int dstep = 0;
            if (shape.size() == 4)
            {
                dstep = shape[2] * shape[3];
            }
            int total = 1;
            for (int c = 0; c < shape.size(); c++)
            {
                total *= shape[c];
            }
            dat.resize(total);
            for (int i = 0; i < total; i++)
            {
                std::vector<int>pos = getPos(i, denominators);
                int c = 0;
                int d = 0;
                int h = 0;
                int w = 0;
                if (pos.size() == 4)
                {
                    c = pos[0];
                    d = pos[1];
                    h = pos[2];
                    w = pos[3];
                }
                if (pos.size() == 3)
                {
                    c = pos[0];
                    h = pos[1];
                    w = pos[2];
                }
                if (pos.size() == 2)
                {
                    h = pos[0];
                    w = pos[1];
                }
                if (pos.size() == 1)
                {
                    w = pos[0];
                }
                float* data = (float*)out.data + d * dstep + c * cstep;
                const float* data2 = data + h * out.w + w;
                dat[i] = *data2;
            }
            return true;
        }
        std::ostream& operator<<(std::ostream& os, const ncnn::Mat& out)
        {
            return printBlob(out,os);
        }
    }

    namespace ocvHelper
    {
        std::ostream& operator<<(std::ostream& os, const cv::dnn::MatShape& shape)
        {
            os << "shape = [";
            for (int i = 0; i < shape.size(); i++)
            {
                if (i == shape.size() - 1)
                {
                    os << shape[i] << "]";
                }
                else
                {
                    os << shape[i] << ", ";
                }
            }
            return os;
        }
        bool operator==(const cv::dnn::MatShape& shape1, const cv::dnn::MatShape& shape2)
        {
            if (shape1.size() != shape2.size())
            {
                return false;
            }
            for (int i = 0; i < shape1.size(); i++)
            {
                if (shape1[i] != shape2[i])
                {
                    return false;
                }
            }
            return true;
        }
        cv::dnn::MatShape getBlobShape(const cv::Mat& blob);

        bool generDnnBlob(cv::Mat& blob, const cv::dnn::MatShape& shape, const OnnxType& type)
        {
            switch (type)
            {
            case OnnxType::onnx_float32:
                blob.create(shape.size(), &shape[0], CV_32F);
                break;
            case OnnxType::onnx_int64:
                blob.create(shape.size(), &shape[0], CV_64FC1);
                break;
            case OnnxType::onnx_int32:
                blob.create(shape.size(), &shape[0], CV_32SC1);
                break;
            default:
                break;
            }
            blob.setTo(0);
            return true;
        }
        bool generPositionBlob(const std::vector<cv::Vec2f>& point_coord,
            const std::vector<float>& point_label,
            cv::Mat& point_coord_blob,
            cv::Mat& point_label_blob,
            const cv::Size& originalImgSize)
        {
            if (point_coord.size() != point_label.size())
            {
                return false;
            }
            int sz1[] = { 1,point_coord.size() + 1,2 };
            point_coord_blob.create(3, sz1, CV_32F);
            for (size_t i = 0; i < point_coord.size() + 1; i++)
            {
                if (i >= point_coord.size())
                {
                    ((float*)point_coord_blob.data)[2 * i] = 0;
                    ((float*)point_coord_blob.data)[2 * i + 1] = 0;
                    break;
                }
                ((float*)point_coord_blob.data)[2 * i] = point_coord[i][0] / originalImgSize.width;
                ((float*)point_coord_blob.data)[2 * i + 1] = point_coord[i][1] / originalImgSize.height;
            }
            int sz2[] = { 1,point_coord.size() + 1,1 };
            point_label_blob.create(3, sz2, CV_32F);
            for (size_t i = 0; i < point_coord.size() + 1; i++)
            {
                if (i >= point_coord.size())
                {
                    ((float*)point_label_blob.data)[i] = -1;
                    break;
                }
                if (point_label[i] > 0)
                {
                    ((float*)point_label_blob.data)[i] = 1;
                }
                else
                {
                    ((float*)point_label_blob.data)[i] = -1;
                }
            }
            return true;
        }
        bool convertNcnnBlobToOpencv(const ncnn::Mat& data, const std::vector<int>& targetShape, cv::Mat& out)
        {
            int srcTotal = 1;
            switch (data.dims)
            {
            case 4:
                srcTotal *= data.c;
            case 3:
                srcTotal *= data.d;
            case 2:
                srcTotal *= data.h;
            case 1:
                srcTotal *= data.w;
            default:
                break;
            }
            int tarTotal = std::accumulate(targetShape.begin(), targetShape.end(), 1, std::multiplies<int>());
            if (tarTotal != srcTotal)
            {
                LOG_ERR_OUT << "tarTotal!=srcTotal";
                return false;
            }
            out.create(targetShape.size(), &targetShape[0], CV_32F);
            memcpy(out.data, data.data, tarTotal * sizeof(float));
            return true;
        }
        bool serializationBlob(const cv::Mat& blob, cv::dnn::MatShape& shape, std::vector<float>& dat)
        {
            shape = getBlobShape(blob);
            int totalcnt = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
            dat.resize(totalcnt);
            memcpy(&dat[0], blob.data, sizeof(float) * totalcnt);
            return true;
        }
        cv::dnn::MatShape getBlobShape(const cv::Mat& blob)
        {
            const int* shape = blob.size.p;
            const int& dim = blob.dims;
            cv::dnn::MatShape shapeRet(dim);
            for (size_t i = 0; i < dim; i++)
            {
                shapeRet[i] = shape[i];
            }
            return shapeRet;
        }
        std::vector<int>getDenominators(const cv::dnn::MatShape& shape)
        {
            std::vector<int>denominators(shape.size(), 1);
            for (int i = shape.size() - 2; i >= 0; i--)
            {
                denominators[i] = denominators[i + 1] * shape[i + 1];
            }
            denominators.back() = 1;
            return denominators;
        }
        std::vector<int>getPos(const int& idx, const std::vector<int>& denominators)
        {
            std::vector<int>pos(denominators.size(), 0);
            pos[0] = idx;
            for (size_t i = 0; i < denominators.size(); i++)
            {
                if (i + 1 < denominators.size())
                {
                    pos[i + 1] = pos[i] % denominators[i];
                }
                pos[i] /= denominators[i];
            }
            return pos;
        }
        void printBlob(const cv::Mat& blob)
        {
            cv::dnn::MatShape shape = getBlobShape(blob);
            std::vector<int>denominators = getDenominators(shape);
            int total = 1;
            std::cout << "shape = ";
            for (int c = 0; c < shape.size(); c++)
            {
                std::cout << shape[c] << ", ";
                total *= shape[c];
            }
            std::cout << std::endl;
            int lineCnt = shape.back();
            bool newLine = false;
            bool newDot = false;
            for (int i = 0; i < total; i++)
            {
                int showFlag = 0;
                std::vector<int>pos = getPos(i, denominators);
                for (size_t j = 0; j < shape.size(); j++)
                {
                    if (shape[j] > 10 && (pos[j] == 4 || pos[j] == (shape[j] - 4)))
                    {
                        showFlag = 1;//...
                        break;
                    }
                    if (shape[j] > 10 && pos[j] > 4 && pos[j] < (shape[j] - 4))
                    {
                        showFlag = 2;//omit
                        break;
                    }
                }
                if (showFlag == 2)
                {
                    continue;
                }
                else if (showFlag == 0)
                {
                    std::cout << ((float*)blob.data)[i] << " ";
                    newLine = false;
                    newDot = false;
                }
                else if (showFlag == 1 && newDot == false)
                {
                    std::cout << " ... ";
                    newDot = true;
                    newLine = false;
                    continue;
                }
                if (i > 0 && pos.back() == (shape.back() - 1) && newLine == false)
                {
                    newLine = true;
                    std::cout << std::endl;
                }
            }
            std::cout << std::endl;
            std::cout << std::endl;
            return;
        }
        void printInt64Blob(const cv::Mat& blob)
        {
            cv::dnn::MatShape shape = getBlobShape(blob);
            int total = 1;
            for (int c = 0; c < shape.size(); c++)
            {
                total *= shape[c];
            }
            int slashN = shape.back();
            for (int i = 0; i < total; i++)
            {
                if (i % slashN == 0)
                {
                    std::cout << std::endl;
                }
                std::cout << ((std::int64_t*)blob.data)[i] << " ";
            }
            std::cout << std::endl;
            return;
        }
        void checkSoftmax(const cv::Mat& blob)
        {
            cv::dnn::MatShape shape = getBlobShape(blob);
            int dim0 = 1;
            int dim1 = shape.back();
            for (int i = 0; i < shape.size() - 1; i++)
            {
                dim0 *= shape[i];
            }
            std::vector<double>checkVALUE(dim0, 0);
            int idx = 0;
            const float* data = (const float*)blob.data;
            for (int i = 0; i < dim0; i++)
            {
                for (int j = 0; j < dim1; j++)
                {
                    checkVALUE[i] += data[idx];
                    idx++;
                }
            }


            return;
        }
        bool readBlobFile(const std::string& path, cv::Mat& blob, const std::optional<cv::dnn::MatShape>optionalShape)
        {
            std::fstream fin(path, std::ios::binary | std::ios::in);
            int dim = 0;
            fin.read((char*)&dim, sizeof(int));
            if (dim <= 0)
            {
                LOG_ERR_OUT << "dim<=0";
                return false;
            }
            std::vector<int>shape(dim);
            fin.read((char*)&shape[0], dim * sizeof(int));
            int totalCnt = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
            if (optionalShape != std::nullopt)
            {
                shape = *optionalShape;
            }
            bool generRet = generDnnBlob(blob, shape);
            if (!generRet)
            {
                LOG_ERR_OUT << "gener blob failed";
                return generRet;
            }
            fin.read((char*)blob.data, totalCnt * sizeof(float));
            return true;
        }
    }
 

    using namespace std;
    using namespace cv;

    enum class OnnxType
    {
        onnx_float32 = 1,
        onnx_uint8 = 2,
        onnx_int8 = 3,
        onnx_uint16 = 4,
        onnx_int16 = 5,
        onnx_int32 = 6,
        onnx_int64 = 7,
        onnx_string = 8,
        onnx_bool = 9,
        onnx_float16 = 10,
        onnx_float64 = 11,
        onnx_uint32 = 12,
        onnx_uint64 = 14,
    };
    bool operator==(const cv::dnn::MatShape& shape1, const cv::dnn::MatShape& shape2)
    {
        if (shape1.size() != shape2.size())
        {
            return false;
        }
        for (int i = 0; i < shape1.size(); i++)
        {
            if (shape1[i] != shape2[i])
            {
                return false;
            }
        }
        return true;
    }
    cv::dnn::MatShape getBlobShape(const cv::Mat& blob);

    bool generPositionBlob(const std::vector<cv::Vec2f>& point_coord,
        const std::vector<float>& point_label,
        cv::Mat& point_coord_blob,
        cv::Mat& point_label_blob,
        const cv::Size& originalImgSize)
    {
        if (point_coord.size() != point_label.size())
        {
            return false;
        }
        int sz1[] = { 1,point_coord.size() + 1,2 };
        point_coord_blob.create(3, sz1, CV_32F);
        for (size_t i = 0; i < point_coord.size() + 1; i++)
        {
            if (i >= point_coord.size())
            {
                ((float*)point_coord_blob.data)[2 * i] = 0;
                ((float*)point_coord_blob.data)[2 * i + 1] = 0;
                break;
            }
            ((float*)point_coord_blob.data)[2 * i] = point_coord[i][0] / originalImgSize.width;
            ((float*)point_coord_blob.data)[2 * i + 1] = point_coord[i][1] / originalImgSize.height;
        }
        int sz2[] = { 1,point_coord.size() + 1,1 };
        point_label_blob.create(3, sz2, CV_32F);
        for (size_t i = 0; i < point_coord.size() + 1; i++)
        {
            if (i >= point_coord.size())
            {
                ((float*)point_label_blob.data)[i] = -1;
                break;
            }
            if (point_label[i] > 0)
            {
                ((float*)point_label_blob.data)[i] = 1;
            }
            else
            {
                ((float*)point_label_blob.data)[i] = -1;
            }
        }
        return true;
    }
    cv::dnn::MatShape getBlobShape(const cv::Mat& blob)
    {
        const int* shape = blob.size.p;
        const int& dim = blob.dims;
        cv::dnn::MatShape shapeRet(dim);
        for (size_t i = 0; i < dim; i++)
        {
            shapeRet[i] = shape[i];
        }
        return shapeRet;
    }
    std::vector<int>getDenominators(const cv::dnn::MatShape& shape)
    {
        std::vector<int>denominators(shape.size(), 1);
        for (int i = shape.size() - 2; i >= 0; i--)
        {
            denominators[i] = denominators[i + 1] * shape[i + 1];
        }
        denominators.back() = 1;
        return denominators;
    }
    std::vector<int>getPos(const int& idx, const std::vector<int>& denominators)
    {
        std::vector<int>pos(denominators.size(), 0);
        pos[0] = idx;
        for (size_t i = 0; i < denominators.size(); i++)
        {
            if (i + 1 < denominators.size())
            {
                pos[i + 1] = pos[i] % denominators[i];
            }
            pos[i] /= denominators[i];
        }
        return pos;
    }
    void printBlob(const cv::Mat& blob)
    {
        cv::dnn::MatShape shape = getBlobShape(blob);
        std::vector<int>denominators = getDenominators(shape);
        int total = 1;
        for (int c = 0; c < shape.size(); c++)
        {
            total *= shape[c];
        }
        int lineCnt = shape.back();
        bool slashN = true;
        int dotCount = 0;
        int dotCountMax = 10;
        for (int i = 0; i < total; i++)
        {
            if (i % lineCnt == 0 && i > 0 && slashN)
            {
                std::cout << std::endl;
            }
            std::vector<int>pos = getPos(i, denominators);
            int showFlag = 0;
            for (size_t j = 0; j < shape.size(); j++)
            {
                if (shape[j] > 10 && (pos[j] == 4 || pos[j] == (shape[j] - 4)))
                {
                    showFlag = 1;
                    break;
                }
                if (shape[j] > 10 && pos[j] > 4 && pos[j] < (shape[j] - 4))
                {
                    showFlag = 2;
                    break;
                }
            }
            if (showFlag == 0)
            {
                dotCount = 0;
                std::cout << ((float*)blob.data)[i] << " ";
                slashN = true;
            }
            else if (showFlag == 1)
            {
                if (dotCount < dotCountMax)
                {
                    std::cout << " ... ";
                    dotCount += 1;
                }
                slashN = true;
            }
            else if (showFlag == 2)
            {
                slashN = false;
            }
        }
        std::cout << std::endl;
        return;
    }
    void printInt64Blob(const cv::Mat& blob)
    {
        cv::dnn::MatShape shape = getBlobShape(blob);
        int total = 1;
        for (int c = 0; c < shape.size(); c++)
        {
            total *= shape[c];
        }
        int slashN = shape.back();
        for (int i = 0; i < total; i++)
        {
            if (i % slashN == 0)
            {
                std::cout << std::endl;
            }
            std::cout << ((std::int64_t*)blob.data)[i] << " ";
        }
        std::cout << std::endl;
        return;
    }
    void checkSoftmax(const cv::Mat& blob)
    {
        cv::dnn::MatShape shape = getBlobShape(blob);
        int dim0 = 1;
        int dim1 = shape.back();
        for (int i = 0; i < shape.size() - 1; i++)
        {
            dim0 *= shape[i];
        }
        std::vector<double>checkVALUE(dim0, 0);
        int idx = 0;
        const float* data = (const float*)blob.data;
        for (int i = 0; i < dim0; i++)
        {
            for (int j = 0; j < dim1; j++)
            {
                checkVALUE[i] += data[idx];
                idx++;
            }
        }


        return;
    }
    int test_dynamic_reshape()
    {
        cv::Mat input;
        ocvHelper::generDnnBlob(input, { 1, 3,2 });
        float* p = (float*)input.data;
        p[0] = 1;
        p[1] = 2;
        p[2] = 3;
        p[3] = 4;
        p[4] = 5;
        p[5] = 6;
        cv::dnn::Net testNet = cv::dnn::readNetFromONNX("D:/repo/colmapThird/test.onnx");
        testNet.setInput(input, "input");
        std::vector<std::string> layersNames = testNet.getLayerNames();
        std::vector<std::string> unconnectedOutLayersNames = testNet.getUnconnectedOutLayersNames();
        std::vector<std::string> outLayersNames = {
        "output","outputReshape" };
        std::vector<cv::Mat> out;
        testNet.forward(out, outLayersNames);
        //printBlob(out[0]);
        //printBlob(out[1]);
        return 0;
    }

}