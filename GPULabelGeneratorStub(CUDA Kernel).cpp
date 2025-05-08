__global__ void generate_labels(char *buffer, int num_labels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_labels) {
        sprintf(buffer + (i * 32), "generated_label_%d", i);
    }
}
