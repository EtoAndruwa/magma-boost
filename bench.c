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

static long double benchmark_simd_minimal(magma_subkeys_256 *ctx,
                                          int iterations,
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
  printf("bytes_processed_simd: %Lf\n", bytes_processed);
  printf("speed_simd: %.2Lf ГБит/с\n", 8 * (speed / 1000000000));

  printf("  Минимальный SIMD тест (8 блоков за операцию):\n");
  printf("    Итераций: %d\n", iterations);
  printf("    Время: %.9Lf сек\n", simd_time);
  printf("    Операций/сек: %.0Lf\n", ops_per_sec);
  printf("    Время на операцию: %.1Lf нс\n", simd_time / iterations * 1e9L);
  return speed;
}

static long double benchmark_scalar_minimal(magma_subkeys *ctx, int iterations,
                                            uint8_t *plaintext,
                                            uint8_t *ciphertext) {
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
  printf("bytes_processed_scalar: %Lf\n", bytes_processed);
  printf("speed_scalar: %.2Lf Гбит/с\n", 8 * (speed / 1000000000));

  printf("  Скалярный тест (1 блок за операцию):\n");
  printf("    Итераций: %d\n", iterations);
  printf("    Время: %.9Lf сек\n", scalar_time);
  printf("    Операций/сек: %.0Lf\n", ops_per_sec);
  printf("    Время на операцию: %.1Lf нс\n", scalar_time / iterations * 1e9L);
  return speed;
}

void test_f_function() {
  printf("\n=== ТЕСТ f и f_simd ФУНКЦИЙ ===\n");

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
  printf("\n=== ТЕСТ ОДНОГО БЛОКА ===\n");

  uint8_t key[32] = {0xff, 0xee, 0xdd, 0xcc, 0xbb, 0xaa, 0x99, 0x88,
                     0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, 0x00,
                     0xf0, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7,
                     0xf8, 0xf9, 0xfa, 0xfb, 0xfc, 0xfd, 0xfe, 0xff};

  magma_subkeys ctx;
  magma_subkeys_256 ctx_256;
  magma_set_key(&ctx, key);
  magma_set_key_256(&ctx_256, key);

  uint8_t plaintext[8] = {0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF};
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
}

void debug_simd_data_loading() {
  printf("\n=== DEBUG ЗАГРУЗКИ ДАННЫХ ===\n");

  uint8_t plaintext[8] = {0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF};

  // 1. Как скалярная версия видит данные
  uint32_t n2_scalar = GETU32_BE(plaintext);
  uint32_t n1_scalar = GETU32_BE(plaintext + 4);

  printf("Скалярная загрузка (GETU32_BE):\n");
  printf("  n2 = 0x%08x (из байтов: ", n2_scalar);
  for (int i = 0; i < 4; i++)
    printf("%02x ", plaintext[i]);
  printf(")\n");

  printf("  n1 = 0x%08x (из байтов: ", n1_scalar);
  for (int i = 4; i < 8; i++)
    printf("%02x ", plaintext[i]);
  printf(")\n\n");

  // 2. Как SIMD версия видит данные
  uint8_t plaintext_256[64];
  for (int i = 0; i < 8; i++) {
    memcpy(plaintext_256 + i * 8, plaintext, 8);
  }

  __m256i block0 = _mm256_load_si256((const __m256i *)(plaintext_256 + 0));
  __m256i block1 = _mm256_load_si256((const __m256i *)(plaintext_256 + 32));

  uint8_t loaded_bytes[64] __attribute__((aligned(32)));
  _mm256_store_si256((__m256i *)(loaded_bytes), block0);
  _mm256_store_si256((__m256i *)(loaded_bytes + 32), block1);

  printf("SIMD загрузка (первый блок):\n");
  printf("  Байты как загружены: ");
  for (int i = 0; i < 8; i++)
    printf("%02x ", loaded_bytes[i]);
  printf("\n");

  // 3. Проверяем shuffle
  const __m256i shuffle_mask =
      _mm256_set_epi8(12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3, 12,
                      13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3);

  __m256i shuffled0 = _mm256_shuffle_epi8(block0, shuffle_mask);
  __m256i shuffled1 = _mm256_shuffle_epi8(block1, shuffle_mask);

  uint8_t shuffled_bytes[64] __attribute__((aligned(32)));
  _mm256_store_si256((__m256i *)(shuffled_bytes), shuffled0);
  _mm256_store_si256((__m256i *)(shuffled_bytes + 32), shuffled1);

  printf("  После shuffle: ");
  for (int i = 0; i < 8; i++)
    printf("%02x ", shuffled_bytes[i]);
  printf("\n");

  // 4. Проверяем разделение на n1 и n2
  const __m256i mask_n2 =
      _mm256_set_epi32(0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000,
                       0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000);

  const __m256i mask_n1 =
      _mm256_set_epi32(0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF,
                       0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF);

  __m256i combined0 = _mm256_permute2x128_si256(shuffled0, shuffled1, 0x20);
  __m256i combined1 = _mm256_permute2x128_si256(shuffled0, shuffled1, 0x31);

  __m256i all_blocks = _mm256_set_m128i(_mm256_extracti128_si256(combined1, 0),
                                        _mm256_extracti128_si256(combined0, 0));

  __m256i n2 = _mm256_and_si256(all_blocks, mask_n2);
  __m256i n1 = _mm256_and_si256(all_blocks, mask_n1);
  n1 = _mm256_srli_epi64(n1, 32);

  uint32_t n1_arr[8] __attribute__((aligned(32))),
      n2_arr[8] __attribute__((aligned(32)));
  _mm256_store_si256((__m256i *)n1_arr, n1);
  _mm256_store_si256((__m256i *)n2_arr, n2);

  printf("\nРазделение на n1 и n2 (первый блок):\n");
  printf("  n2[0] = 0x%08x (ожидается 0x%08x)\n", n2_arr[0], n2_scalar);
  printf("  n1[0] = 0x%08x (ожидается 0x%08x)\n", n1_arr[0], n1_scalar);
}

int main() {
  test_f_function();
  test_single_block_encryption();

  // Тестовый ключ из ГОСТ
  uint8_t key[32] = {0xff, 0xee, 0xdd, 0xcc, 0xbb, 0xaa, 0x99, 0x88,
                     0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, 0x00,
                     0xf0, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7,
                     0xf8, 0xf9, 0xfa, 0xfb, 0xfc, 0xfd, 0xfe, 0xff};

  magma_subkeys ctx;
  magma_subkeys_256 ctx_256;
  magma_set_key(&ctx, key);
  magma_set_key_256(&ctx_256, key);

  srand(time(NULL));

  printf("\nТест производительности:\n");

  uint8_t plaintext[8];
  uint8_t ciphertext[8];

  int iterations = 10000000;

  long double speed_scalar =
      benchmark_scalar_minimal(&ctx, iterations, plaintext, ciphertext);

  uint8_t plaintext_256[64] __attribute__((aligned(32)));
  uint8_t ciphertext_256[64] __attribute__((aligned(32)));

  for (int i = 0; i < 8; i++) {
    memcpy(plaintext_256 + i * 8, plaintext, 8);
  }

  long double speed_simd = benchmark_simd_minimal(
      &ctx_256, iterations, plaintext_256, ciphertext_256);

  printf("\nSpeedup: %LF\n", speed_simd / speed_scalar);
  return 0;
}
