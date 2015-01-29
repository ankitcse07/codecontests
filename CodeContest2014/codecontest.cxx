
#include <immintrin.h>
#include "sys/time.h"
#include "stdio.h"
#include <string.h>
#include <stdlib.h>
#include <unordered_map>
#include <vector>
#include <stack>
#include <queue>
#include <omp.h>
#include <unistd.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>

using namespace std;

#ifdef _OPENMP 
#define NUM_THREADS 8 
#else
#define NUM_THREADS 1
#endif


#define NEW_EMPL 1
#define FRIENDS 2
#define SME 3

long ** vertexAdjacencyMap;
char *vertexTypeMap;

unordered_map<long, int> newEmployeeMap;
unordered_map<long, int> smeMap;
unordered_map<long, int>** nodeConntMap;
char *mappedBuffer;


__m128i newline;
__m128i dashChar;

double wtime(void);

double totalProcTime =0;

struct GraphEdge
{
    long vertex1;
    long vertex2;
};

vector <GraphEdge *> graphEdges[NUM_THREADS];
long *vertexAdjancenyListSize;

long largestThreadNodeID[NUM_THREADS];
long largestNodeID = 0;

void GetLargestNodeIDThread(long long start,long long end, long long size)
{
    int threadID = omp_get_thread_num();
    // Skip first line
    while(end < size)
    {
        if((mappedBuffer[end] != '\n') && mappedBuffer[end] != '}')
            end++;
        else
            break;
    }
    if(!start  || mappedBuffer[start - 1] != '\n')
    {
        while(start <= end)
        {
            if(mappedBuffer[start] == '\n')
            {
                start++;
                break;
            }
            start++;
        }
    }
    if(start > end)
        return;

    char *token;
    long vertex1, vertex2;
    int index = 0;

    char * saveptr = new char[64];
    bool isVertex1, isVertex2;
    __m128i byteVector;
    isVertex1 = true;
    isVertex2 = false;
    int startVertexByte;
    int maskBits;

    index = start;
    startVertexByte = start;
    token = mappedBuffer;

    long largest = 0;
    GraphEdge *edgePool;
    long approxPoolSize = (end - start)/20 + 1; // ID--ID
    long edgePoolIndex = 0;
    edgePool = new GraphEdge[approxPoolSize];
    graphEdges[threadID].reserve(approxPoolSize);

    while((index + 16) <= end)
    {
        byteVector = _mm_loadu_si128((__m128i*)(token + index));
        if(isVertex1)
            maskBits = _mm_cmpistri(byteVector, dashChar, _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_LEAST_SIGNIFICANT);
        else
            maskBits = _mm_cmpistri(byteVector, newline, _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_LEAST_SIGNIFICANT);


        if(maskBits != 16)
        {
            if(isVertex1)
            {
                vertex1 = atol(token + startVertexByte);
                startVertexByte = index + maskBits + 2;
                index = startVertexByte;
                isVertex1 = false;
            }
            else
            {
                vertex2 = atol(token + startVertexByte);
                startVertexByte = index + maskBits + 1;
                index = startVertexByte;
                isVertex1 = true;
                if(vertex1 > vertex2)
                {
                    if(vertex1 > largest)
                        largest = vertex1;
                }
                else
                {
                    if(vertex2 > largest)
                        largest = vertex2;
                }
                
                if(edgePoolIndex >= approxPoolSize)
                {
                    approxPoolSize = (end - start)/20 + 1; // ID--ID
                    edgePoolIndex = 0;
                    edgePool = new GraphEdge[approxPoolSize];
                    graphEdges[threadID].reserve(approxPoolSize);
                }
                
                if(edgePoolIndex < approxPoolSize)
                {
                   edgePool[edgePoolIndex].vertex1 = vertex1;
                   edgePool[edgePoolIndex].vertex2 = vertex2;
                   graphEdges[threadID].push_back(edgePool + edgePoolIndex);
                   edgePoolIndex++;
                }
            }
        }
        else
            index = index + 16;
    }

    if(startVertexByte <= end)
    {
        char tmpStr[32];
        strncpy(tmpStr, mappedBuffer + startVertexByte, (end - startVertexByte + 1));
        tmpStr[end - startVertexByte + 1] = 0;

        token = tmpStr;
        if(!isVertex1)
        {
            token = strtok_r (token, "\n", &saveptr);
            vertex2 = atol(token);
            if(vertex1 > vertex2)
            {
                if(vertex1 > largest)
                    largest = vertex1;
            }
            else
            {
                if(vertex2 > largest)
                    largest = vertex2;
            }
            if(edgePoolIndex >= approxPoolSize)
            {
                approxPoolSize = (end - start)/20 + 1; // ID--ID
                edgePoolIndex = 0;
                edgePool = new GraphEdge[approxPoolSize];
                graphEdges[threadID].reserve(approxPoolSize);
            }

            if(edgePoolIndex < approxPoolSize)
            {
                edgePool[edgePoolIndex].vertex1 = vertex1;
                edgePool[edgePoolIndex].vertex2 = vertex2;
                graphEdges[threadID].push_back(edgePool + edgePoolIndex);
                edgePoolIndex++;
            }
            token = strtok_r (NULL, "-", &saveptr);
        }
        else
            token = strtok_r ( token, "-", &saveptr);
        while(token && (token[0] != '}'))
        {
            vertex1 = atol(token);
            token = strtok_r(NULL, "-\n", &saveptr);
            vertex2 = atol(token);
            if(vertex1 > vertex2)
            {
                if(vertex1 > largest)
                    largest = vertex1;
            }
            else
            {
                if(vertex2 > largest)
                    largest = vertex2;
            }
            if(edgePoolIndex >= approxPoolSize)
            {
                approxPoolSize = (end - start)/20 + 1; // ID--ID
                edgePoolIndex = 0;
                edgePool = new GraphEdge[approxPoolSize];
                graphEdges[threadID].reserve(approxPoolSize);
            }
            if(edgePoolIndex < approxPoolSize)
            {
                edgePool[edgePoolIndex].vertex1 = vertex1;
                edgePool[edgePoolIndex].vertex2 = vertex2;
                graphEdges[threadID].push_back(edgePool + edgePoolIndex);
                edgePoolIndex++;
            }
            token = strtok_r(NULL, "-", &saveptr);
        }
    }
    largestThreadNodeID[omp_get_thread_num()] = largest;
}

long *vertexConnThread[NUM_THREADS];

void AddEdge(long vertex1, long vertex2)
{
    int threadID = omp_get_thread_num();
    long * adjancyList1;
    long * adjancyList2;

    char vertex1Type, vertex2Type;

    vertex1Type = FRIENDS;
    vertex2Type = FRIENDS;

#ifdef _DEBUG
    printf("Graph edge %ld--%ld\n", vertex1, vertex2);
#endif
    if(vertexTypeMap[vertex1] == 0)
    {
        if(newEmployeeMap.find(vertex1) != newEmployeeMap.end())
            vertex1Type = NEW_EMPL;
        else if(smeMap.find(vertex1) != smeMap.end())
            vertex1Type = SME;
        vertexTypeMap[vertex1] = vertex1Type;
    }

    if(vertexTypeMap[vertex2] == 0)
    {
        if(newEmployeeMap.find(vertex2) != newEmployeeMap.end())
            vertex2Type = NEW_EMPL;
        else if(smeMap.find(vertex2) != smeMap.end())
            vertex2Type = SME;
        vertexTypeMap[vertex2] = vertex2Type;
    }


    adjancyList1 = vertexAdjacencyMap[vertex1];
    adjancyList2 = vertexAdjacencyMap[vertex2];
    adjancyList1[vertexConnThread[threadID][vertex1]] = vertex2;
    vertexConnThread[threadID][vertex1] += 1;
    adjancyList2[vertexConnThread[threadID][vertex2]] = vertex1;
    vertexConnThread[threadID][vertex2] += 1;
}

void CreateGraphFromEdges(int threadID)
{
    vector <GraphEdge*>::iterator itr;
    GraphEdge *edge;

    for(itr = graphEdges[threadID].begin(); itr != graphEdges[threadID].end(); itr++)
    {
        edge = *itr;
        AddEdge(edge->vertex1, edge->vertex2);
    }
}


void AllocateMemoryAndAssignSize(long start, long end)
{
    long startIndex = 0, countStored;
    long index;
    if(!start)
        start++;

    long long totalSizeForAdjList = 0;
    for(index = start; index <= end; index++)
    {
        startIndex = 0;
        vertexAdjancenyListSize[index] = 0;
        for(int tid = 0 ; tid < NUM_THREADS; tid++)
        {
            countStored = vertexConnThread[tid][index];
            vertexAdjancenyListSize[index] += countStored;
            vertexConnThread[tid][index] = startIndex;
            startIndex += countStored;
        }
        totalSizeForAdjList += vertexAdjancenyListSize[index];
    }
    long *allocatePool = new long[totalSizeForAdjList];
    for(index = start; index <= end; index++)
    {
        vertexAdjacencyMap[index] = allocatePool;
        allocatePool += vertexAdjancenyListSize[index];
    }
}

void DetermineVertexAdjConnForThread()
{
    int index = omp_get_thread_num();
    vector <GraphEdge*>::iterator itr;
    GraphEdge *edge;
    vertexConnThread[index] = new long[largestNodeID + 1];
    memset(vertexConnThread[index], 0,  sizeof(long)*(largestNodeID + 1));

    for(itr = graphEdges[index].begin(); itr != graphEdges[index].end(); itr++)
    {
        edge = *itr;
        vertexConnThread[index][edge->vertex1]++;
        vertexConnThread[index][edge->vertex2]++;
    }
}

void CreateGraph(const char *name)
{
    newline = _mm_set1_epi8('\n');
    dashChar = _mm_set1_epi8('-');

    struct stat fileInfo;
    int largeFD;

    largeFD = open (name, O_RDONLY);

    if(largeFD == -1)
    {
        printf("Fatal: input_graph missing. Exiting ! \n");
        exit(-1);
    }
    if (fstat (largeFD, &fileInfo) == -1) {
        perror ("fstat");
        return;
    }

    mappedBuffer = (char *) mmap (0, fileInfo.st_size, PROT_READ, MAP_SHARED, largeFD, 0);

    long index = 0;
    long long size = fileInfo.st_size;
    long long segment = size/NUM_THREADS;

#pragma omp parallel default(none) shared(segment) shared(size) private(index) num_threads(NUM_THREADS)
#pragma omp for schedule(static, 1)
    for(index = 0; index < NUM_THREADS; ++index)
    { 
        GetLargestNodeIDThread(index*segment, (index != (NUM_THREADS - 1))  ? index*segment + segment -1 : size - 1, size);
    }

    for(index = 0 ; index < NUM_THREADS; ++index)
    {
        if(largestThreadNodeID[index] > largestNodeID)
            largestNodeID = largestThreadNodeID[index];
    }

    vertexAdjancenyListSize = new long[largestNodeID + 1];

    vector <GraphEdge*>::iterator itr;
    GraphEdge *edge;


#pragma omp parallel default(none) private(index) num_threads(NUM_THREADS)
#pragma omp for schedule(static, 1)
    for (index = 0; index < NUM_THREADS; index++)
    {
        DetermineVertexAdjConnForThread();
    }

    vertexAdjacencyMap = new long* [largestNodeID + 1];

    size = largestNodeID + 1;
    segment = size/NUM_THREADS;
#pragma omp parallel default(none) private(index) shared(size) shared(segment) num_threads(NUM_THREADS)
#pragma omp for schedule(static, 1)
    for (index = 0; index < NUM_THREADS; index++)
    {
        AllocateMemoryAndAssignSize(index*segment, (index != (NUM_THREADS - 1))  ? index*segment + segment -1 : size - 1);
    }

    vertexTypeMap = new char[largestNodeID + 1];
    memset(vertexTypeMap, 0, largestNodeID + 1);
#pragma omp parallel default(none) private(index) num_threads(NUM_THREADS)
#pragma omp for schedule(static, 1)
    for(index = 0; index < NUM_THREADS; ++index)
    { 
        CreateGraphFromEdges(index);
    }

    for(index = 0; index < NUM_THREADS; ++index)
    {
        delete [] vertexConnThread[index];
    }

    nodeConntMap = new unordered_map<long, int>*[largestNodeID + 1];
    memset(nodeConntMap, 0, sizeof(nodeConntMap)*(largestNodeID + 1));
    close (largeFD);
    munmap (mappedBuffer, fileInfo.st_size);
}

void PopulateSmeList(const char* name)
{
    FILE *fp = fopen(name, "r");
    if(fp == NULL)
    {
        printf("Missing file %s\n", name);
        exit(-1);
    }

    char line[256];

    while(fgets(line, 256, fp) != NULL)
        smeMap[atol(line)] = 1;
    fclose(fp);
}


void PopulateNewEmployeeList(const char* name)
{
    FILE *fp = fopen(name, "r");
    if(fp == NULL)
    {
        printf("Missing file %s\n", name);
        exit(-1);
    }

    char line[256];

    long empID ;
    while(fgets(line, 256, fp) != NULL)
    {
        empID  = atol(line);
        newEmployeeMap[empID] = 1;
    }

    fclose(fp);
}

FILE* outputFP;
void dumpOutputFile(int start, int size)
{
    double startTime = wtime();
    unordered_map<long, int>::iterator newEmpMapIter;
    unordered_map<long, int>::iterator newEmpMapIterLast;
    unordered_map<long, int>::iterator newEmpMapIterStart;

    newEmpMapIter = newEmployeeMap.begin();
    newEmpMapIterStart = newEmpMapIter;

    std::advance(newEmpMapIterStart, start);

    newEmpMapIterLast = newEmpMapIterStart;
    std::advance(newEmpMapIterLast, size);
    long * adjancyList;
    unordered_map<long,int>*connDest;

    const unsigned long long BUFFER_SIZE =64ULL*4096ULL;
    char *localBuffer = new char [BUFFER_SIZE + 512];
    char *localBufferPtr;
    int charWritten;
    long long totalBytesInBuffer;

    unordered_map<long, unordered_map<long, int>*>::iterator nodeConntMapIter;
    long vecIter;
    int index;

    localBufferPtr = localBuffer;
    totalBytesInBuffer = 0;

    int fdno = fileno(outputFP);
    for(newEmpMapIter=newEmpMapIterStart; newEmpMapIter != newEmpMapIterLast; ++newEmpMapIter)
    {
        adjancyList  = vertexAdjacencyMap[newEmpMapIter->first];
        if(adjancyList != NULL)
        {
            unordered_map<long, int> removeDuplicate;
            char nodeType;
            for(index = 0 ; index < vertexAdjancenyListSize[newEmpMapIter->first]; ++index)
            {
                vecIter = adjancyList[index];
                nodeType = vertexTypeMap[vecIter];
                if(nodeType == SME)
                {
                    if(removeDuplicate.find(vecIter) == removeDuplicate.end())
                    {
                        charWritten = sprintf(localBufferPtr, "%ld--%ld\n", newEmpMapIter->first, vecIter);
                        totalBytesInBuffer += charWritten;
                        removeDuplicate[vecIter] = 1;

                        if(totalBytesInBuffer < BUFFER_SIZE)
                            localBufferPtr += charWritten;
                        else
                        {
                            localBufferPtr = localBuffer;
                            #pragma omp critical
                            {
                                fwrite(localBuffer, 1, totalBytesInBuffer, outputFP);
                            }
                            totalBytesInBuffer = 0;
                        }
                    }
                }
                else
                {
                    connDest = nodeConntMap[vecIter];
                    if(connDest != NULL)
                    {
                        unordered_map<long, int>::iterator destNodeIter;

                        for(destNodeIter = connDest->begin(); destNodeIter != connDest->end(); ++destNodeIter)
                        {
                    
                            if(removeDuplicate.find(destNodeIter->first) == removeDuplicate.end())
                            {

                                charWritten = sprintf(localBufferPtr, "%ld--%ld--%ld\n", newEmpMapIter->first, vecIter, destNodeIter->first);
                                //charWritten = sprintf(localBufferPtr, "%ld--%ld\n", newEmpMapIter->first, destNodeIter->first);
                                totalBytesInBuffer += charWritten;
                                removeDuplicate[destNodeIter->first] = 1;
                                if(totalBytesInBuffer < BUFFER_SIZE)
                                    localBufferPtr += charWritten;
                                else
                                {
                                    localBufferPtr = localBuffer;
                                    #pragma omp critical
                                    {
                                        fwrite(localBuffer, 1, totalBytesInBuffer, outputFP);
                                    }
                                    totalBytesInBuffer = 0;
                                }
                            }
                        }
                    }
                }
            }
        }

    }

    if(totalBytesInBuffer > 0)
    {
        #pragma omp critical
        {
            fwrite(localBuffer, 1, totalBytesInBuffer, outputFP);
        }
    }
}

void findPathNonRecursive()
{
    unordered_map<long, int>::iterator smeMapIter;

    stack<long> travStack;
    long* adjancyList;
    long queNode;
    char queNodeType;
    double startime;

    unordered_map<long, int>*connDest;
    for(smeMapIter=smeMap.begin(); smeMapIter != smeMap.end(); ++smeMapIter)
    {
        if(nodeConntMap[smeMapIter->first] == NULL)
            travStack.push(smeMapIter->first);

        while(!travStack.empty())
        {
            queNode = travStack.top();
            travStack.pop();
                
            queNodeType = vertexTypeMap[queNode];
            if(queNodeType == NEW_EMPL)
                continue;

            if(queNodeType == SME)
            {
                connDest = nodeConntMap[queNode];
                if(connDest == NULL)
                {
                    connDest = new unordered_map<long, int>;
                    nodeConntMap[queNode] = connDest;
                }
                (*connDest)[queNode] = 1;
            }

            adjancyList = vertexAdjacencyMap[queNode];

            if(adjancyList != NULL)
            {
                int index;
                long vecIter;
                char nodeType;
                for(index= 0; index < vertexAdjancenyListSize[queNode]; ++index)
                {
                    vecIter = adjancyList[index]; 
                    nodeType = vertexTypeMap[vecIter];

                    if(queNodeType == SME)
                    {
                        if(nodeType == SME)
                            continue;

                        if(nodeType != NEW_EMPL)
                        {
                            connDest = nodeConntMap[vecIter];
                            if(connDest == NULL)
                            {
                                nodeConntMap[vecIter] = nodeConntMap[queNode];
                                travStack.push(vecIter);
                            }
                        }
                    }
                    else if(queNodeType != NEW_EMPL)
                    {
                        if(nodeType == SME)
                        {
                            connDest = nodeConntMap[queNode];
                            (*connDest)[vecIter] = 1;
                            //nodeConntMap[vecIter] = connDest;
                        }
                        else if(nodeType != NEW_EMPL)
                        {
                            connDest = nodeConntMap[vecIter];
                            if(connDest == NULL)
                            {
                                nodeConntMap[vecIter] = nodeConntMap[queNode];
                                travStack.push(vecIter);
                            }
                        }
                    }
                }
            }
        }
    }
}

double initTime();
void recordTime();


void FindAllPaths()
{
    outputFP = fopen("output", "wb");

  initTime();
   findPathNonRecursive();
  recordTime();
    
    int index = 0;
    unsigned int totalEmpList = newEmployeeMap.size();
    unsigned int segment = totalEmpList/NUM_THREADS;

  initTime();
#pragma omp parallel default(none) shared(segment) shared(totalEmpList) private(index) num_threads(NUM_THREADS)
#pragma omp for schedule(static, 1)
    for(index = 0; index < NUM_THREADS; ++index)
    { 
        dumpOutputFile(index*segment, (index != (NUM_THREADS -1)) ? segment: (totalEmpList - (index*segment)));
    }
    fclose(outputFP);
  recordTime();
}

FILE *outputNodeFP;
int myMain()
{
  PopulateSmeList("sme_list");
  PopulateNewEmployeeList("new_employee_list");
  initTime();
  CreateGraph("input_graph");
  recordTime();
  FindAllPaths();
}


/******************************************************************
 * Do Not Modify This Section
 ******************************************************************/
double wtime(void)
{
  double sec;
  struct timeval tv;
  gettimeofday(&tv,NULL);
  sec = tv.tv_sec + tv.tv_usec/1000000.0;
  return sec;
}

double initTime()
{
  static double startTime = 0; 
  double lastStartTime = startTime;
  startTime = wtime();
  return lastStartTime;
}

void recordTime()
{
  printf("#### Recorded Time:%f\n",wtime()-initTime());
}

int main()
{
  initTime();
  myMain();
  recordTime();
  return 0;
}
/*******************************************************************/
