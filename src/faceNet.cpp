#include "faceNet.h"


int FaceNetClassifier::m_classCount = 0;

FaceNetClassifier::FaceNetClassifier
(Logger gLogger, DataType dtype, const string uffFile, const string engineFile, int batchSize, bool serializeEngine,
        float knownPersonThreshold, int maxFacesPerScene, int frameWidth, int frameHeight) {
    /// ONNG
    cout << "======================= LOAD ONNG AND TEST ONE VECTOR=========================" << endl;
    string indexPath	= "index_onng_1m_1tr50";
    string queryFile	= "foo.tsv";

    onng_index = new NGT::Index(indexPath);

    property.dimension		= 128;
    property.objectType		= NGT::ObjectSpace::ObjectType::Float;
    property.distanceType	= NGT::Index::Property::DistanceType::DistanceTypeCosine;
    onng_index->getProperty(property);
    ifstream		is(queryFile);
    string		line;
    vector<float>	query;
    getline(is, line);
    stringstream	linestream(line);
    while (!linestream.eof()) {
	  float value;
	  linestream >> value;
	  //std::cout << value << endl  ;
	  query.push_back(value);
	}

    query.resize(property.dimension);
    cout << "Query : ";
    for (size_t i = 0; i < 5; i++) {
        cout << static_cast<float>(query[i]) << " ";
    }
    cout << "...";

    NGT::SearchQuery		sc(query);
    NGT::ObjectDistances	objects;
    sc.setResults(&objects);
    sc.setSize(10);
    sc.setEpsilon(0.1);
	
    onng_index->search(sc);
    cout << endl << "Rank\tID\tDistance" << std::showbase << endl;
    for (size_t i = 0; i < objects.size(); i++) {
	cout << i + 1 << "\t" << objects[i].id << "\t" << objects[i].distance << "\t: ";
	NGT::ObjectSpace &objectSpace = onng_index->getObjectSpace();
	uint8_t *object = static_cast<uint8_t*>(objectSpace.getObject(objects[i].id));
	for (size_t idx = 0; idx < 5; idx++) {
	  cout << static_cast<int>(object[idx]) << " ";
	}
     cout << "..." << endl;
     }

  cout << "======================= LOAD ONNG SUCCESSED=========================" << endl;
    ///////////////////////////////// CLEAR TEXT FILE ///////////////////////////////
    std::ofstream outfile ("test.txt", ios::out | ios::trunc);
    outfile<<"";
    outfile.close();
    /////////////////////////////////////////////////////////////////////
    face_count = 0;
    m_INPUT_C = static_cast<const int>(3);
    m_INPUT_H = static_cast<const int>(160);
    m_INPUT_W = static_cast<const int>(160);
    m_frameWidth = static_cast<const int>(frameWidth);
    m_frameHeight = static_cast<const int>(frameHeight);
    m_gLogger = gLogger;
    m_dtype = dtype;
    m_uffFile = static_cast<const string>(uffFile);
    m_engineFile = static_cast<const string>(engineFile);
    m_batchSize = batchSize;
    m_serializeEngine = serializeEngine;
    m_maxFacesPerScene = maxFacesPerScene;
    m_croppedFaces.reserve(maxFacesPerScene);
    m_embeddings.reserve(128);
    m_knownPersonThresh = knownPersonThreshold;
    camera_id = "0";
    // load engine from .engine file or create new engine
    this->createOrLoadEngine();
    cout << "currentDateTime()===================="  << this->currentDateTime() << endl;
}

const std::string FaceNetClassifier::currentDateTime() {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%Y-%m-%d %X", &tstruct);
    return buf;
}

void FaceNetClassifier::createOrLoadEngine() {
    if(fileExists(m_engineFile)) {
        std::vector<char> trtModelStream_;
        size_t size{ 0 };

        std::ifstream file(m_engineFile, std::ios::binary);
        if (file.good())
        {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream_.resize(size);
            std::cout << "size" << trtModelStream_.size() << std::endl;
            file.read(trtModelStream_.data(), size);
            file.close();
        }
        // std::cout << "size" << size;
        IRuntime* runtime = createInferRuntime(m_gLogger);
        assert(runtime != nullptr);
        m_engine = runtime->deserializeCudaEngine(trtModelStream_.data(), size, nullptr);
        std::cout << std::endl;
    }
    else {
        IBuilder *builder = createInferBuilder(m_gLogger);
        INetworkDefinition *network = builder->createNetwork();
        IUffParser *parser = createUffParser();
        parser->registerInput("input", DimsCHW(160, 160, 3), UffInputOrder::kNHWC);
        parser->registerOutput("embeddings");

        if (!parser->parse(m_uffFile.c_str(), *network, m_dtype))
        {
            cout << "Failed to parse UFF\n";
            builder->destroy();
            parser->destroy();
            network->destroy();
            throw std::exception();
        }

        /* build engine */
        if (m_dtype == DataType::kHALF)
        {
            builder->setFp16Mode(true);
        }
        else if (m_dtype == DataType::kINT8) {
            builder->setInt8Mode(true);
            // ToDo
            //builder->setInt8Calibrator()
        }
        builder->setMaxBatchSize(m_batchSize);
        builder->setMaxWorkspaceSize(1<<30);
        // strict will force selected datatype, even when another was faster
        //builder->setStrictTypeConstraints(true);
        // Disable DLA, because many layers are still not supported
        // and this causes additional latency.
        //builder->allowGPUFallback(true);
        //builder->setDefaultDeviceType(DeviceType::kDLA);
        //builder->setDLACore(1);
        m_engine = builder->buildCudaEngine(*network);

        /* serialize engine and write to file */
        if(m_serializeEngine) {
            ofstream planFile;
            planFile.open(m_engineFile);
            IHostMemory *serializedEngine = m_engine->serialize();
            planFile.write((char *) serializedEngine->data(), serializedEngine->size());
            planFile.close();
        }

        /* break down */
        builder->destroy();
        parser->destroy();
        network->destroy();
    }
    m_context = m_engine->createExecutionContext();
}


void FaceNetClassifier::getCroppedFacesAndAlign(cv::Mat frame, std::vector<struct Bbox> outputBbox) {
    for(vector<struct Bbox>::iterator it=outputBbox.begin(); it!=outputBbox.end();it++){
        if((*it).exist){
            cv::Rect facePos(cv::Point((*it).y1, (*it).x1), cv::Point((*it).y2, (*it).x2));
            cv::Mat tempCrop = frame(facePos);
            struct CroppedFace currFace;
            cv::resize(tempCrop, currFace.faceMat, cv::Size(160, 160), 0, 0, cv::INTER_CUBIC);
            currFace.x1 = it->x1;
            currFace.y1 = it->y1;
            currFace.x2 = it->x2;
            currFace.y2 = it->y2;            
            m_croppedFaces.push_back(currFace);
        }
    }
    //ToDo align
}

void FaceNetClassifier::preprocessFaces() {
    // preprocess according to facenet training and flatten for input to runtime engine
    for (int i = 0; i < m_croppedFaces.size(); i++) {
        //mean and std
        cv::cvtColor(m_croppedFaces[i].faceMat, m_croppedFaces[i].faceMat, cv::COLOR_RGB2BGR);
        cv::Mat temp = m_croppedFaces[i].faceMat.reshape(1, m_croppedFaces[i].faceMat.rows * 3);
        cv::Mat mean3;
        cv::Mat stddev3;
        cv::meanStdDev(temp, mean3, stddev3);

        double mean_pxl = mean3.at<double>(0);
        double stddev_pxl = stddev3.at<double>(0);
        cv::Mat image2;
        m_croppedFaces[i].faceMat.convertTo(image2, CV_64FC1);
        m_croppedFaces[i].faceMat = image2;
        // fix by peererror
        cv::Mat mat(4, 1, CV_64FC1);
		mat.at <double>(0, 0) = mean_pxl;
		mat.at <double>(1, 0) = mean_pxl;
		mat.at <double>(2, 0) = mean_pxl;
		mat.at <double>(3, 0) = 0;
        m_croppedFaces[i].faceMat = m_croppedFaces[i].faceMat - mat;
        // end fix
        m_croppedFaces[i].faceMat = m_croppedFaces[i].faceMat / stddev_pxl;
        m_croppedFaces[i].faceMat.convertTo(image2, CV_32FC3);
        m_croppedFaces[i].faceMat = image2;
    }
}


void FaceNetClassifier::doInference(float* inputData, float* output) {
    int size_of_single_input = 3 * 160 * 160 * sizeof(float);
    int size_of_single_output = 128 * sizeof(float);
    int inputIndex = m_engine->getBindingIndex("input");
    int outputIndex = m_engine->getBindingIndex("embeddings");

    void* buffers[2];

    cudaMalloc(&buffers[inputIndex], m_batchSize * size_of_single_input);
    cudaMalloc(&buffers[outputIndex], m_batchSize * size_of_single_output);

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // copy data to GPU and execute
    CHECK(cudaMemcpyAsync(buffers[inputIndex], inputData, m_batchSize * size_of_single_input, cudaMemcpyHostToDevice, stream));
    m_context->enqueue(m_batchSize, &buffers[0], stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], m_batchSize * size_of_single_output, cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    
    // Release the stream and the buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}


void FaceNetClassifier::forwardAddFace(cv::Mat image, std::vector<struct Bbox> outputBbox,
        const string className) {
    
    //cv::resize(image, image, cv::Size(1280, 720), 0, 0, cv::INTER_CUBIC);
    getCroppedFacesAndAlign(image, outputBbox);
    if(!m_croppedFaces.empty()) {
        preprocessFaces();
        doInference((float*)m_croppedFaces[0].faceMat.ptr<float>(0), m_output);
        struct KnownID person;
        person.className = className;
        person.classNumber = m_classCount;
        person.embeddedFace.insert(person.embeddedFace.begin(), m_output, m_output+128);
        m_knownFaces.push_back(person);
        m_classCount++;
    }
    m_croppedFaces.clear();
}

void FaceNetClassifier::forward(cv::Mat frame, std::vector<struct Bbox> outputBbox) {
    getCroppedFacesAndAlign(frame, outputBbox); // ToDo align faces according to points
    preprocessFaces();
    for(int i = 0; i < m_croppedFaces.size(); i++) {
        doInference((float*)m_croppedFaces[i].faceMat.ptr<float>(0), m_output);
        m_embeddings.insert(m_embeddings.end(), m_output, m_output+128);
    }
}

void FaceNetClassifier::featureMatching(cv::Mat &image) {
    std::ofstream outfile ("test.txt",fstream::app);
    
    for(int i = 0; i < (m_embeddings.size()/128); i++) {
        std::string list_id;
        double minDistance =  m_knownPersonThresh;
        float currDistance = 0.;
        int winner = -1;

        std:vector<float> currEmbedding(128);
        std::copy_n(m_embeddings.begin()+(i*128), 128, currEmbedding.begin());
	///////// ONNG SEARCH	

	NGT::SearchQuery		sc(currEmbedding);
	NGT::ObjectDistances	objects;
	sc.setResults(&objects);
	sc.setSize(5);
	sc.setEpsilon(0.3);
	onng_index->search(sc);
        
	//out << endl << "Rank\tID\tDistance" << std::showbase << endl;
	for (size_t i = 0; i < objects.size(); i++) {
		if  ( objects[0].distance > m_knownPersonThresh ) break; 
		list_id.append( std::to_string(objects[i].id) ) ;
                list_id.append( "," ) ;
		//	cout << i + 1 << "\t" << objects[i].id << "\t" << objects[i].distance << "\t: ";
			/// put name
			//NGT::ObjectSpace &objectSpace = onng_index->getObjectSpace();
			//float *object = static_cast<float*>(objectSpace.getObject(objects[i].id));
			//for (size_t idx = 0; idx < 5; idx++) {
			//  cout << object[idx] << " ";
			//}
		//	cout << endl;
	}

	currEmbedding.clear();
        float fontScaler = static_cast<float>(m_croppedFaces[i].x2 - m_croppedFaces[i].x1)/static_cast<float>(m_frameWidth);
        cv::rectangle(image, cv::Point(m_croppedFaces[i].y1, m_croppedFaces[i].x1), cv::Point(m_croppedFaces[i].y2, m_croppedFaces[i].x2), 
                        cv::Scalar(0,0,255), 2,8,0);
        
	if ( objects[0].distance <= m_knownPersonThresh) {
           face_count = face_count + 1;
  	    outfile << face_count << " " << currentDateTime() << " " << camera_id << " " << list_id << " " << std::endl;
            /////////////// save image
            cv::Mat cropped_image = image(cv::Rect(cv::Point(m_croppedFaces[i].y1, m_croppedFaces[i].x1), cv::Point(m_croppedFaces[i].y2, m_croppedFaces[i].x2)));
            string filePath = "../send_imgs/";
            filePath.append(std::to_string( face_count ));
            filePath.append(".jpg");
            cv::imwrite( filePath, cropped_image );
            //////////////////////////
	    cv::putText(image, std::to_string(objects[0].id) , cv::Point(m_croppedFaces[i].y1+2, m_croppedFaces[i].x2-3),
		    cv::FONT_HERSHEY_DUPLEX, 0.1 + 2*fontScaler,  cv::Scalar(0,255,0,255), 1);
	}
    }
    outfile.close();
}

void FaceNetClassifier::addNewFace(cv::Mat &image, std::vector<struct Bbox> outputBbox) {
    std::cout << "Adding new person...\nPlease make sure there is only one face in the current frame.\n"
              << "What's your name? ";
    string newName;
    std::cin >> newName;
    std::cout << "Hi " << newName << ", you will be added to the database.\n";
    forwardAddFace(image, outputBbox, newName);
    string filePath = "../imgs/";
    filePath.append(newName);
    filePath.append(".jpg");
    cv::imwrite(filePath, image);
}

void FaceNetClassifier::resetVariables() {
    m_embeddings.clear();
    m_croppedFaces.clear();
}

FaceNetClassifier::~FaceNetClassifier() {
    // this leads to segfault 
    // this->m_engine->destroy();
    // this->m_context->destroy();
    // std::cout << "FaceNet was destructed" << std::endl;
}


// HELPER FUNCTIONS
// Computes the distance between two std::vectors
float vectors_distance(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<double>	auxiliary;
    std::transform (a.begin(), a.end(), b.begin(), std::back_inserter(auxiliary),//
                    [](float element1, float element2) {return pow((element1-element2),2);});
    auxiliary.shrink_to_fit();
    float loopSum = 0.;
    for(auto it=auxiliary.begin(); it!=auxiliary.end(); ++it) loopSum += *it;

    return  std::sqrt(loopSum);
} 



inline unsigned int elementSize(nvinfer1::DataType t)
{
    switch (t)
    {
        case nvinfer1::DataType::kINT32:
            // Fallthrough, same as kFLOAT
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT8: return 1;
    }
    assert(0);
    return 0;
}
