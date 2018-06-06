void upsweep(__local int* temp, const int gthid, const int lthid, const int group_size, const int block_size, unsigned int* offset);
void downsweep(__local int* temp, const int gthid, const int lthid, const int group_size, const int block_size, unsigned int* offset);

__kernel void scan_init(__global int *input, __global int *output, __local int *temp, __global int *sums)
{
	int gid = get_group_id(0);
	int gthid = get_global_id(0);
	int lthid = get_local_id(0);
	int group_size = get_local_size(0);
	int block_size = 2 * group_size;// x2 weil halb so viele threads wie items

	int lower_bound = gid * block_size;
	int n = (gid + 1) * block_size;

	unsigned int offset[1] = { 1 };

	temp[2 * lthid] = input[2 * gthid]; // load input into shared memory
	temp[2 * lthid + 1] = input[2 * gthid + 1];

	upsweep(temp, gthid, lthid, group_size, block_size, offset);

	// delete the first element
	if (lthid == 0) {
		sums[gid] = temp[block_size - 1];
		temp[block_size - 1] = 0;
	}

	downsweep(temp, gthid, lthid, group_size, block_size, offset);

	barrier(CLK_LOCAL_MEM_FENCE);

	// write the results
	output[2 * gthid] = temp[2 * lthid];
	output[2 * gthid + 1] = temp[2 * lthid + 1];

}

void upsweep(__local int* temp, const int gthid,const int lthid, const int group_size, const int block_size, unsigned int* offset)
{
	for (int d = block_size >> 1; d > 0; d >>= 1) 
	{
		barrier(CLK_LOCAL_MEM_FENCE);

		if (lthid < d)
		{
			int ai = *offset * (2 * lthid + 1) - 1;
			int bi = *offset * (2 * lthid + 2) - 1;

			temp[bi] += temp[ai];
		}

		*offset *= 2;
	}
}

void downsweep(__local int* temp, const int gthid, const int lthid, const int group_size, const int block_size, unsigned int* offset)
{
	for (int d = 1; d < block_size; d *= 2) 
	{
		*offset >>= 1;
		barrier(CLK_LOCAL_MEM_FENCE);

		if (lthid < d) 
		{
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
	int gthid = get_global_id(0);
	int lthid = get_local_id(0);
	int group_size = get_local_size(0); //1024

	int block_size = group_size * 2; // 2048
	int gid = get_group_id(0); // 2048 groups

	int offset = block_size * gid; //e.g. 3 * 2048 = 6188 each group writes into an area 2048 apart

	int address = 2 * lthid + offset;
	int address2 = 2 * lthid + offset + 1;

	int sum = sums[gid];
	if (gid == 8)
	{
		printf("sum = %d\n", sums[gid]);
	}
	data[address] += sum;
	data[address2] += sum;
}

__kernel void even(__global int *input, __global int *output)
{
	int gthid = get_global_id(0);
	output[gthid] = input[gthid] % 2 == 0;
}

__kernel void odd(__global int *input, __global int *output)
{
	int gthid = get_global_id(0);
	output[gthid] = input[gthid] % 2 == 1;
}

__kernel void lesser500(__global int *input, __global int *output)
{
	int gthid = get_global_id(0);
	output[gthid] = input[gthid] < 500;
}

__kernel void compact(__global int *input, __global int *addresses, __global int *output, int output_size)
{
	int gthid = get_global_id(0);
	int address = addresses[gthid];

	if (gthid == get_global_size(0) - 1) {
		if (address < output_size) {
			output[address] = input[gthid];
		}
	}
	else if (address != addresses[gthid + 1]) {
		output[address] = input[gthid];
	}
}
