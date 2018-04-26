#define _XOPEN_SOURCE 600
#include "doomdev.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/ioctl.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#define WIDTH 640
#define HEIGHT 480

static void fill_rects(int fd, const struct doomdev_fill_rect *rects, int rects_num) {
	struct doomdev_surf_ioctl_fill_rects fill;
	int res;
	do {
		fill.rects_ptr = (uintptr_t)rects;
		fill.rects_num = rects_num < 0xffff ? rects_num : 0xffff;
		res = ioctl(fd, DOOMDEV_SURF_IOCTL_FILL_RECTS, &fill);
		if (res < 0) {
			perror("fill_rects");
			exit(1);
		}
		if (res == 0 || res > fill.rects_num) {
			fprintf(stderr, "fill_rects: WTF %d\n", res);
			exit(1);
		}
		rects_num -= res;
		rects += res;
	} while (rects_num);
}

static void copy_rects(int dstfd, int srcfd, const struct doomdev_copy_rect *rects, int rects_num) {
	struct doomdev_surf_ioctl_copy_rects copy = {(uintptr_t)rects, srcfd, rects_num};
	int res;
	do {
		res = ioctl(dstfd, DOOMDEV_SURF_IOCTL_COPY_RECTS, &copy);
		if (res < 0) {
			perror("copy_rects");
			exit(1);
		}
		if (res == 0 || res > copy.rects_num) {
			fprintf(stderr, "copy_rects: WTF %d\n", res);
			exit(1);
		}
		copy.rects_num -= res;
		rects -= res;
		copy.rects_ptr = (uintptr_t)rects;
	} while (copy.rects_num);
}

static void do_read(int fd, uint8_t *ptr, size_t num) {
	int pos = 0;
	while (num) {
		ssize_t res = pread(fd, ptr, num, pos);
		if (res < 0) {
			perror("read");
			exit(1);
		}
		if (res == 0 || res > num) {
			fprintf(stderr, "read: WTF %zd\n", res);
			exit(1);
		}
		num -= res;
		ptr += res;
		pos += res;
	}
}

static void validate(int fd, uint8_t *data) {
	static uint8_t tmp[WIDTH*HEIGHT];
	do_read(fd, tmp, sizeof tmp);
	if (memcmp(tmp, data, sizeof tmp)) {
		int pos;
		for (pos = 0; pos < sizeof tmp; pos++)
			if (data[pos] != tmp[pos])
				break;
		fprintf(stderr, "MISMATCH %x %x\n", pos % WIDTH, pos / WIDTH);
		exit(1);
	}
}

int main(int argc, char **argv) {
	const char *fname = "/dev/doom0";
	if (argc >= 2)
		fname = argv[1];
	int dfd = open(fname, O_RDWR);
	if (dfd < 0) {
		perror("open");
		return 1;
	}
	struct doomdev_ioctl_create_surface cs = {WIDTH, HEIGHT};
	int s1fd = ioctl(dfd, DOOMDEV_IOCTL_CREATE_SURFACE, &cs);
	if (s1fd < 0) {
		perror("create_surface");
		return 1;
	}
	int s2fd = ioctl(dfd, DOOMDEV_IOCTL_CREATE_SURFACE, &cs);
	if (s2fd < 0) {
		perror("create_surface");
		return 1;
	}
	srand(time(0));
	int reps = 100000;
	if (argc >= 3)
		reps = atoi(argv[2]);
	static uint8_t sim1[HEIGHT][WIDTH];
	static uint8_t sim2[HEIGHT][WIDTH];
	static struct doomdev_fill_rect rects1[WIDTH * HEIGHT];
	static struct doomdev_fill_rect rects2[WIDTH * HEIGHT];
	for (int y = 0; y < HEIGHT; y++) {
		for (int x = 0; x < WIDTH; x++) {
			int idx = x + y * WIDTH;
			uint8_t col1 = rand();
			uint8_t col2 = rand();
			rects1[idx] = (struct doomdev_fill_rect){x, y, 1, 1, col1};
			rects2[idx] = (struct doomdev_fill_rect){x, y, 1, 1, col2};
			sim1[y][x] = col1;
			sim2[y][x] = col2;
		}
	}
	fill_rects(s1fd, rects1, WIDTH * HEIGHT);
	fill_rects(s2fd, rects2, WIDTH * HEIGHT);
	for (int i = 0; i < reps; i++) {
reroll:;
		int x1 = rand() % (WIDTH + 1);
		int x2 = rand() % (WIDTH + 1);
		int y1 = rand() % (HEIGHT + 1);
		int y2 = rand() % (HEIGHT + 1);
		int w = rand() % (WIDTH + 1);
		int h = rand() % (HEIGHT + 1);
		if (x1 + w > WIDTH)
			goto reroll;
		if (x2 + w > WIDTH)
			goto reroll;
		if (y1 + h > HEIGHT)
			goto reroll;
		if (y2 + h > HEIGHT)
			goto reroll;
		struct doomdev_copy_rect r = {x1, y1, x2, y2, w, h};
		//printf("%x %x %x %x %x %x\n", x1, y1, x2, y2, w, h);
		copy_rects(s1fd, s2fd, &r, 1);
		for (int dy = 0; dy < h; dy++)
			for (int dx = 0; dx < w; dx++)
				sim1[y1+dy][x1+dx] = sim2[y2+dy][x2+dx];
		validate(s1fd, (void*)sim1);
	}
	return 0;
}
