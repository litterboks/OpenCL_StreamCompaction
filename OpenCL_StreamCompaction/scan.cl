void upsweep(__local int* temp, const int thid, const int lthid, const int group_size, const int block_size, unsigned int* offset);
void downsweep(__local int* temp, const int thid, const int lthid, const int group_size, const int block_size, unsigned int* offset);

__kernel void scan_init(__global int *input, __global int *output, __local int *temp, __global int *sums)
{
	int thid = get_global_id(0);
	int gid = get_group_id(0);
	int lthid = get_local_id(0);
	int group_size = get_local_size(0);
	int block_size = 2 * group_size;// x2 weil halb so viele threads wie items

	int lower_bound = gid * block_size;
	int n = (gid + 1) * block_size;

	unsigned int offset[1] = { 1 };

	temp[2 * lthid] = input[2 * thid]; // load input into shared memory
	temp[2 * lthid + 1] = input[2 * thid + 1];

	upsweep(temp, thid, lthid, group_size, block_size, offset);

	// delete the first element
	if (lthid == 0) {
		sums[gid] = temp[block_size - 1];
		temp[block_size - 1] = 0;
	}

	downsweep(temp, thid, lthid, group_size, block_size, offset);

	barrier(CLK_LOCAL_MEM_FENCE);

	// write the results
	output[2 * thid] = temp[2 * lthid];
	output[2 * thid + 1] = temp[2 * lthid + 1];

}

void upsweep(__local int* temp, const int thid,const int lthid, const int group_size, const int block_size, unsigned int* offset)
{
	for (int d = block_size >> 1; d > 0; d >>= 1) {
		barrier(CLK_LOCAL_MEM_FENCE);

		if (lthid < d) {
			int ai = *offset * (2 * lthid + 1) - 1;
			int bi = *offset * (2 * lthid + 2) - 1;

			temp[bi] += temp[ai];
		}

		*offset *= 2;
	}
}

void downsweep(__local int* temp, const int thid, const int lthid, const int group_size, const int block_size, unsigned int* offset)
{
	for (int d = 1; d < block_size; d *= 2) {
		*offset >>= 1;
		barrier(CLK_LOCAL_MEM_FENCE);

		if (lthid < d) {
			//D
			int ai = *offset * (2 * lthid + 1) - 1;
			int bi = *offset * (2 * lthid + 2) - 1;

			float t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
}

__kernel void add_sums(__global int *data, __global int *sums)
{
	int thid = get_global_id(0);
	int lthid = get_local_id(0);
	int group_size = get_local_size(0);
	int block_size = group_size * 2;
	int gid = get_group_id(0);

	int offset = block_size * gid;

	int address = 2 * lthid + offset;
	int address2 = 2 * lthid + offset + 1;

	int sum = sums[get_group_id(0)];

	data[address] += sum;
	data[address2] += sum;
}