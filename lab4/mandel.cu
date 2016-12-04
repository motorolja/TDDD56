// Mandelbrot explorer, based on my old Julia demo plus parts of Nicolas Melot's Lab 1 code.
// CPU only! Your task: Rewrite for CUDA! Test and evaluate performance.

// Compile with:
// gcc interactiveMandelbrot.cpp -shared-libgcc -lstdc++-static  -o interactiveMandelbrot -lglut -lGL
// or
// g++ interactiveMandelbrot.cpp -o interactiveMandelbrot -lglut -lGL

// Your CUDA version should compile with something like
// nvcc -lglut -lGL interactiveMandelbrotCUDA.cu -o interactiveMandelbrotCUDA

// Preliminary version 2014-11-30
// Cleaned a bit more 2014-12-01
// Corrected the missing glRasterPos2i 2014-12-03

#include <GL/glut.h>
#include <GL/gl.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Image data
unsigned char	*pixels = NULL;
unsigned char	*gpu_bitmap = NULL;
int	 gImageWidth, gImageHeight;

// Init image data
void initBitmap(int width, int height)
{
	if (pixels) free(pixels);
	pixels = (unsigned char *)malloc(width * height * 4);
	gImageWidth = width;
	gImageHeight = height;
}

#define DIM 512
#define PIXELS DIM*DIM
#define PIXEL_DATA PIXELS*4

// Select precision here! float or double!
#define MYFLOAT float

// User controlled parameters
int maxiter = 20;
MYFLOAT offsetx = -200, offsety = 0, zoom = 0;
MYFLOAT scale = 1.5;

// Complex number class
struct cuComplex
{
  MYFLOAT   r;
  MYFLOAT   i;

  __device__
  cuComplex( MYFLOAT a, MYFLOAT b ) : r(a), i(b)  {}

  __device__
  float magnitude2( void )
  {
    return r * r + i * i;
  }

  __device__
  cuComplex operator*(const cuComplex& a)
  {
    return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
  }

  __device__
  cuComplex operator+(const cuComplex& a)
  {
    return cuComplex(r+a.r, i+a.i);
  }
};


// Only called in the CUDA kernel
__device__
int mandelbrot( int x, int y)
{
  float gscale = 1.5;
  const int maxiter = 200;

  MYFLOAT jx = gscale * (MYFLOAT)(DIM/2 - x)/(DIM/2);
  MYFLOAT jy = gscale * (MYFLOAT)(DIM/2 - y)/(DIM/2);

  cuComplex c(jx, jy);
  cuComplex a(jx, jy);

  int i = 0;
  for (i=0; i<maxiter; i++)
    {
      a = a * a + c;
      if (a.magnitude2() > 1000)
        return i;
    }

  return i;
}

// Entry point for CUDA calls
__global__
void computeFractal( unsigned char *ptr, float mouse_x, float mouse_y)
{
  int const maxiter = 200;
  // compute indexes for gpu threads
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int indexes = col * DIM + row;

  // just a sanity check, should never occur
  //  if (row >= DIM || col >= DIM || indexes >= PIXELS)
  //    {
  //      return;
  //    }
  // calculate the value at that position
  int fractalValue = mandelbrot( col, row);

  // Colorize it
  int red = 255 * fractalValue/maxiter;
  if (red > 255) red = 255 - red;
  int green = 255 * fractalValue*4/maxiter;
  if (green > 255) green = 255 - green;
  int blue = 255 * fractalValue*20/maxiter;
  if (blue > 255) blue = 255 - blue;

  ptr[indexes*4 + 0] = red;
  ptr[indexes*4 + 1] = green;
  ptr[indexes*4 + 2] = blue;

  ptr[indexes*4 + 3] = 255;
}

char print_help = 0;
// Yuck, GLUT text is old junk that should be avoided... but it will have to do
static void print_str(void *font, const char *string)
{
	int i;

	for (i = 0; string[i]; i++)
		glutBitmapCharacter(font, string[i]);
}

void PrintHelp()
{
	if (print_help)
    {
      glPushMatrix();
      glLoadIdentity();
      glOrtho(-0.5, 639.5, -0.5, 479.5, -1.0, 1.0);

      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
      glColor4f(0.f, 0.f, 0.5f, 0.5f);
      glRecti(40, 40, 600, 440);

      glColor3f(1.f, 1.f, 1.f);
      glRasterPos2i(300, 420);
      print_str(GLUT_BITMAP_HELVETICA_18, "Help");

      glRasterPos2i(60, 390);
      print_str(GLUT_BITMAP_HELVETICA_18, "h - Toggle Help");
      glRasterPos2i(60, 300);
      print_str(GLUT_BITMAP_HELVETICA_18, "Left click + drag - move picture");
      glRasterPos2i(60, 270);
      print_str(GLUT_BITMAP_HELVETICA_18,
                "Right click + drag up/down - unzoom/zoom");
      glRasterPos2i(60, 240);
      print_str(GLUT_BITMAP_HELVETICA_18, "+ - Increase max. iterations by 32");
      glRasterPos2i(60, 210);
      print_str(GLUT_BITMAP_HELVETICA_18, "- - Decrease max. iterations by 32");
      glRasterPos2i(0, 0);

      glDisable(GL_BLEND);

      glPopMatrix();
    }
}


// Compute fractal and display image
void Draw()
{
  // Allocation of memory for CUDA should be done once and not every iteration
  dim3 dimBlock(DIM, DIM); // might need tuning
  dim3 dimGrid(64, 64); // might need tuning

  // CUDA events
  cudaEvent_t e1, e2;
  cudaEventCreate(&e1);
  cudaEventCreate(&e2);
  cudaEventRecord(e1,0);
  cudaEventSynchronize(e1);

  // Call CUDA with the kernel with the picture + mouse offset
	computeFractal <<<dimGrid, dimBlock>>> (gpu_bitmap,0.0,0.0);
  // wait for the whole image to be calculated
  cudaThreadSynchronize();

  // Synchronize and get the time between e1 - e2
  cudaEventRecord(e2,0);
  cudaEventSynchronize(e2);
  float time_elapsed;
  cudaEventElapsedTime(&time_elapsed, e1, e2);
  printf("Time elapsed(CUDA): %f ms\n", time_elapsed);

  // clean up
  cudaEventDestroy(e1);
  cudaEventDestroy(e2);

  // copy back the result from the GPU calculations
  cudaMemcpy(pixels, gpu_bitmap, DIM*DIM*4, cudaMemcpyDeviceToHost);

  // Dump the whole picture onto the screen. (Old-style OpenGL but without lots of geometry that doesn't matter so much.)
	glClearColor( 0.0, 0.0, 0.0, 1.0 );
	glClear( GL_COLOR_BUFFER_BIT );
	glDrawPixels( gImageWidth, gImageHeight, GL_RGBA, GL_UNSIGNED_BYTE, pixels );

	if (print_help)
		PrintHelp();

	glutSwapBuffers();
}

char explore = 1;

static void Reshape(int width, int height)
{
	glViewport(0, 0, width, height);
	glLoadIdentity();
	glOrtho(-0.5f, width - 0.5f, -0.5f, height - 0.5f, -1.f, 1.f);
	initBitmap(width, height);

	glutPostRedisplay();
}

int mouse_x, mouse_y, mouse_btn;

// Mouse down
static void mouse_button(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
    {
      // Record start position
      mouse_x = x;
      mouse_y = y;
      mouse_btn = button;
    }
}

// Drag mouse
static void mouse_motion(int x, int y)
{
	if (mouse_btn == 0)
    // Ordinary mouse button - move
    {
      offsetx += (x - mouse_x)*scale;
      mouse_x = x;
      offsety += (mouse_y - y)*scale;
      mouse_y = y;

      glutPostRedisplay();
    }
	else
    // Alt mouse button - scale
    {
      scale *= pow(1.1, y - mouse_y);
      mouse_y = y;
      glutPostRedisplay();
    }
}

void KeyboardProc(unsigned char key, int x, int y)
{
	switch (key)
    {
    case 27: /* Escape key */
    case 'q':
    case 'Q':
      exit(0);
      break;
    case '+':
      maxiter += maxiter < 1024 - 32 ? 32 : 0;
      break;
    case '-':
      maxiter -= maxiter > 0 + 32 ? 32 : 0;
      break;
    case 'h':
      print_help = !print_help;
      break;
    }
	glutPostRedisplay();
}

// Main program, inits
int main( int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
	glutInitWindowSize( DIM, DIM );
	glutCreateWindow("Mandelbrot explorer (GPU)");
	glutDisplayFunc(Draw);
	glutMouseFunc(mouse_button);
	glutMotionFunc(mouse_motion);
	glutKeyboardFunc(KeyboardProc);
	glutReshapeFunc(Reshape);

	initBitmap(DIM, DIM);

  // Allocate GPU memory, picture has DIM x DIM size + 4 color values
  cudaMalloc(&gpu_bitmap,PIXEL_DATA);

	glutMainLoop();
  // Free memory
  cudaFree(gpu_bitmap);
}
