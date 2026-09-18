#include "all_testcases.h"
#include "lib.h"

void run_builtin_command(string const &cmd) {
  if (cmd == "ls") {
    cout << "Listing all testcases..." << endl;
    for (Testcase const &testcase : all_testcases) {
      cout << "- " << testcase.name;
      cout << ", [";
      for (string const &tag : testcase.tags) {
        cout << tag << ", ";
      }
      cout << "\b\b], ";
      cout << testcase.description << endl;
    }
  } else {
    cerr << "Unknown command: " << cmd << endl;
  }
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage:\n");
    printf("  %s <testcase-name>, or\n", argv[0]);
    printf("  %s !<builtin-command>, or\n", argv[0]);
    printf("  %s +<tag>\n", argv[0]);
    return 1;
  }
  string testcase_name = argv[1];
  if (!testcase_name.empty() && testcase_name[0] == '!') {
    // Execute a builtin command
    run_builtin_command(testcase_name.substr(1));
    return 0;
  }

  vector<pair<Testcase &, vector<Subcase::RunResult>>> results;
  for (Testcase &testcase : all_testcases) {
    bool should_run = false;
    if (!testcase_name.empty() && testcase_name[0] == '+') {
      should_run = testcase_name == "+" || std::count(testcase.tags.begin(),
                                                      testcase.tags.end(),
                                                      testcase_name.substr(1));
    } else {
      should_run = testcase.name == testcase_name;
    }
    if (should_run) {
      printf("==================================================\n");
      printf("==================================================\n");
      printf("Running testcase %s...\n", testcase.name.c_str());
      vector<Subcase::RunResult> cur_results = testcase.run();
      results.push_back({testcase, cur_results});
    }
  }

  // Print the summary
  printf("==================================================\n");
  printf("==================================================\n");
  printf("Summary:\n");
  for (auto const &[testcase, subcase_results] : results) {
    bool is_first_line = true;
    int max_subcase_name_len = 0;
    for (Subcase::RunResult const &subcase_result : subcase_results) {
      max_subcase_name_len = std::max(
          max_subcase_name_len, (int)subcase_result.subcase_name.length());
    }
    for (Subcase::RunResult const &subcase_result : subcase_results) {
      string subcase_name =
          string(max_subcase_name_len - subcase_result.subcase_name.length(),
                 ' ') +
          subcase_result.subcase_name;
      if (subcase_name != "") {
        subcase_name += ", ";
      }
      printf("  %30s | %s%s, %6.2f ms%s%s\n",
             is_first_line ? testcase.name.c_str() : "",
             subcase_name.c_str(),
             subcase_result.is_passed ? "\e[97;42;1m PASSED \e[0m"
                                      : "\e[97;41;1m FAILED \e[0m",
             subcase_result.avg_time_ms,
             subcase_result.msg.length() == 0 ? "" : ", ",
             subcase_result.msg.c_str());
      is_first_line = false;
    }
  }
  return 0;
}