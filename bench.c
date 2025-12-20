#include <immintrin.h>
#include <stdalign.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <x86intrin.h>

#include "magma.h"
#include "magma_simd.h"

long double scalar_time = 0;
long double simd_time = 0;

// Высокоточный таймер
static uint64_t get_nanoseconds() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

void print_magma_blocks(const uint8_t *data, size_t total_bytes) {
  size_t num_blocks = total_bytes / 8;

  printf("\n");
  for (size_t block = 0; block < num_blocks; block++) {
    printf("Блок %2zu: ", block);
    for (size_t i = 0; i < 8; i++) {
      printf("%02x ", data[block * 8 + i]);
    }
    printf("\n");
  }
}

void print_m256i_hex(__m256i value, const char *name) {
  printf("%s: ", name);

  alignas(32) uint32_t values[8];
  _mm256_store_si256((__m256i *)values, value);

  for (int i = 0; i < 8; i++) {
    printf("0x%08x ", values[i]);
  }
  printf("\n");
}

static void fill_random(uint8_t *data, size_t size) {
  for (size_t i = 0; i < size; i++)
    data[i] = rand() & 0xFF;
}

static long double benchmark_simd(magma_subkeys_256 *ctx, int iterations,
                                  uint8_t *plaintext_256,
                                  uint8_t *ciphertext_256) {
  uint64_t start = get_nanoseconds();
  for (int iter = 0; iter < iterations; iter++)
    magma_encrypt_8blocks(ctx, ciphertext_256, plaintext_256);
  uint64_t end = get_nanoseconds();

  print_magma_blocks(ciphertext_256, 64);

  simd_time = (end - start) / 1e9;
  long double ops_per_sec = iterations / simd_time;
  long double bytes_processed = 64 * iterations;
  long double speed = bytes_processed / simd_time;
  speed = 8 * (speed / 1000000000);

  printf("  Минимальный SIMD тест (8 блоков за операцию):\n");
  printf("    Итераций: %d\n", iterations);
  printf("    Время: %.9Lf сек\n", simd_time);
  printf("    Операций/сек: %.0Lf\n", ops_per_sec);
  printf("    Время на операцию: %.1Lf нс\n", simd_time / iterations * 1e9L);
  printf("    Скорость SIMD реализации: %.2Lf ГБит/с\n\n", speed);
  return speed;
}

static long double benchmark_scalar(magma_subkeys *ctx, int iterations,
                                    uint8_t *plaintext, uint8_t *ciphertext) {
  fill_random(plaintext, 8);

  iterations = iterations * 8;

  uint64_t start = get_nanoseconds();
  for (int iter = 0; iter < iterations; iter++)
    magma_encrypt_scalar(ctx, ciphertext, plaintext);
  uint64_t end = get_nanoseconds();

  print_magma_blocks(ciphertext, 8);

  scalar_time = (end - start) / 1e9;
  long double ops_per_sec = iterations / scalar_time;
  long double bytes_processed = 8 * iterations;
  long double speed = bytes_processed / scalar_time;
  speed = 8 * (speed / 1000000000);

  printf("  Скалярный тест (1 блок за операцию):\n");
  printf("    Итераций: %d\n", iterations);
  printf("    Время: %.9Lf сек\n", scalar_time);
  printf("    Операций/сек: %.0Lf\n", ops_per_sec);
  printf("    Время на операцию: %.1Lf нс\n", scalar_time / iterations * 1e9L);
  printf("    Скорость скалярной реализации: %.2Lf Гбит/с\n", speed);
  return speed;
}

void test_f_function() {
  printf("=== Тест f и f_simd функций ===\n");

  // Тестовые значения
  uint32_t test_values[] = {0x01234567, 0x89ABCDEF, 0xFEDCBA98, 0x76543210,
                            0x00000000, 0xFFFFFFFF, 0x12345678, 0x9ABCDEF0};

  for (int i = 0; i < 8; i++) {
    uint32_t scalar_result = f(test_values[i]);

    // Создаем SIMD вектор с одним значением, повторенным 8 раз
    __m256i simd_input = _mm256_set1_epi32(test_values[i]);
    __m256i simd_result_vec = f_simd(simd_input);

    // Извлекаем первое значение
    uint32_t simd_result_arr[8];
    _mm256_store_si256((__m256i *)simd_result_arr, simd_result_vec);
    uint32_t simd_result = simd_result_arr[0];

    printf("Тест %d: 0x%08x\n", i, test_values[i]);
    printf("  Скалярный f: 0x%08x\n", scalar_result);
    printf("  SIMD f_simd: 0x%08x\n", simd_result);

    if (scalar_result != simd_result) {
      printf("  ✗ НЕ СОВПАДАЕТ!\n");

      // Детальный debug
      uint8_t *bytes = (uint8_t *)&test_values[i];
      printf("    Байты: %02x %02x %02x %02x\n", bytes[0], bytes[1], bytes[2],
             bytes[3]);
      printf("    pi87[%02x]=0x%08x, pi65[%02x]=0x%08x\n", bytes[3],
             pi87[bytes[3]], bytes[2], pi65[bytes[2]]);
      printf("    pi43[%02x]=0x%08x, pi21[%02x]=0x%08x\n", bytes[1],
             pi43[bytes[1]], bytes[0], pi21[bytes[0]]);
    } else {
      printf("  ✓ СОВПАДАЕТ\n");
    }
    printf("\n");
  }
}

void test_single_block_encryption() {
  printf("\n=== Тест одного блока ===\n");

  uint8_t key[32] = {0xff, 0xee, 0xdd, 0xcc, 0xbb, 0xaa, 0x99, 0x88,
                     0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, 0x00,
                     0xf7, 0xf6, 0xf5, 0xf4, 0xf3, 0xf2, 0xf1, 0xf0,
                     0xff, 0xfe, 0xfd, 0xfc, 0xfb, 0xfa, 0xf9, 0xf8};

  magma_subkeys ctx;
  magma_subkeys_256 ctx_256;
  magma_set_key(&ctx, key);
  magma_set_key_256(&ctx_256, key);

  uint8_t plaintext[8] = {0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10};
  uint8_t ciphertext_scalar[8] __attribute__((aligned(32)));
  uint8_t ciphertext_simd[64] __attribute__((aligned(32)));

  printf("Plaintext: ");
  for (int i = 0; i < 8; i++)
    printf("%02x ", plaintext[i]);
  printf("\n\n");

  magma_encrypt_scalar(&ctx, ciphertext_scalar, plaintext);
  printf("Скалярный ciphertext: ");
  for (int i = 0; i < 8; i++)
    printf("%02x ", ciphertext_scalar[i]);
  printf("\n");

  uint8_t plaintext_256[64];
  for (int i = 0; i < 8; i++) {
    memcpy(plaintext_256 + i * 8, plaintext, 8);
  }

  magma_encrypt_8blocks(&ctx_256, ciphertext_simd, plaintext_256);

  printf("SIMD ciphertext (первый блок): ");
  for (int i = 0; i < 8; i++)
    printf("%02x ", ciphertext_simd[i]);
  printf("\n\n");

  // Сравнение
  int match = 1;
  for (int i = 0; i < 8; i++) {
    if (ciphertext_scalar[i] != ciphertext_simd[i]) {
      match = 0;
      printf("Ошибка на позиции %d: скаляр=%02x, SIMD=%02x\n", i,
             ciphertext_scalar[i], ciphertext_simd[i]);
    }
  }

  if (match) {
    printf("✓ Результаты совпадают!\n");
  } else {
    printf("✗ Результаты НЕ совпадают\n");
  }
  printf("=== Тест одного блока ===\n");
}

int main() {
  test_f_function();
  test_single_block_encryption();

  // Тестовый ключ из ГОСТ
  uint8_t key[32] = {0xff, 0xee, 0xdd, 0xcc, 0xbb, 0xaa, 0x99, 0x88,
                     0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, 0x00,
                     0xf7, 0xf6, 0xf5, 0xf4, 0xf3, 0xf2, 0xf1, 0xf0,
                     0xff, 0xfe, 0xfd, 0xfc, 0xfb, 0xfa, 0xf9, 0xf8};

  magma_subkeys ctx;
  magma_subkeys_256 ctx_256;
  magma_set_key(&ctx, key);
  magma_set_key_256(&ctx_256, key);

  srand(time(NULL));

  printf("\n=== Тест производительности ===\n");

  uint8_t plaintext[8];
  uint8_t ciphertext[8];

  int iterations = 10000000;

  long double speed_scalar =
      benchmark_scalar(&ctx, iterations, plaintext, ciphertext);

  uint8_t plaintext_256[64] __attribute__((aligned(32)));
  uint8_t ciphertext_256[64] __attribute__((aligned(32)));

  for (int i = 0; i < 8; i++) {
    memcpy(plaintext_256 + i * 8, plaintext, 8);
  }

  long double speed_simd =
      benchmark_simd(&ctx_256, iterations, plaintext_256, ciphertext_256);

  printf("Ускорение относительно реализации из linux-kernel: %LF\n",
         speed_simd / speed_scalar);
  printf("Ускорение относительно реализации из gost-engine: %LF\n",
         speed_simd / 0.22);
  return 0;
}
