import re
inloop, found = False, False

new_lines = {}
start_line = 0
purge_match, profiler_match = False, False
purge_regex = re.compile(r'purge_audit.*threshold_space = (\d+) GB, used_space = (\d+)')
#with open ('/Users/srinara2/Downloads/showtech(2).out', 'r') as myfile: # Open show_tech.txt for reading text data.

print (inloop)
for line_num, line in enumerate(open('/Users/srinara2/Downloads/showtech(2).out', 'rt',errors="ignore")):
        if inloop:
            # Match the first section following the section I care about and stop reading

            if 'Session With High Open Cursors  - under threshold' in line:

                inloop = False

            if "MNT        PROFILER_PROFILED" in line:
                found = True
                print(inloop)

                start_line = line_num
                new_lines[start_line] = {}
                new_lines[start_line]['line'] = [x.strip() for x in line.split() if x.strip()]
                print(new_lines[start_line])
                print(new_lines)


            elif found and not line.strip():
                new_lines[start_line]['end'] = line_num - 1
                print('---')

                print(new_lines)
                found = False
            elif found:
                new_lines[start_line]['line'].extend([x.strip() for x in line.split() if x.strip()])
                print(new_lines)

        # Only start looping through MnT logic once entering the section that matters
        elif 'Objects With Stale Stats  - threshold exceeded' in line:
            inloop = True

        # Only fire if this is causing space issues
        elif purge_regex.search(line):
            purge = purge_regex.search(line)
            threshold = int(purge.group(1))
            used = int(purge.group(2))

            #result_list.debug('Used Space = {}'.format(used / threshold))
            if used / threshold > .75:
                purge_match = True

for l, v in new_lines.items():
        # result_list.debug("Start = {}, End = {}, Line = {}".format(l, v['end'], v['line']))
        x = (v['line'][3])
        print(x)

        if v['line'][1] == 'PROFILER_PROFILED' and int(v['line'][3]) >= 250000:
            profiler_match = True

            print(x)
            print('--+++--')
            #ise_sig.line_matches.append(borg3.result.LineValue(int(l), int(v['end'])))

if purge_match and profiler_match:


        ise_sig_text = """
                    The MnT Database on this node is currently running at {}% of the data retention threshold and there is evidence that the PROFILER_PROFILED table is not being appropriately purged.
                    This will eventually fill up the MnT database. Please refer to the bug link below for patched versions of ISE. In the short term the MnT Database can be reset if the MnT data
                    on this node is no longer needed or logged externally.<br>
                    Reference: <a href="https://scripts.cisco.com/app/quicker_cdets/?bug=CSCvm35110" target="_blank" rel="noopener noreferer">CSCvm35110</a>
                """.format(int(used / threshold * 100))
        ise_sig_external_text = """
                    The MnT Database on this node is currently running at {} of the data retention threshold and there is evidence that the PROFILER_PROFILED table is not being appropriately purged.
                    This will eventually fill up the MnT database. Please refer to the bug link below for patched versions of ISE. In the short term the MnT Database can be reset if the MnT data
                    on this node is no longer needed or logged externally.<br>
                    Reference: <a href="https://bst.cloudapps.cisco.com/bugsearch/bug/CSCvm35110" target="_blank" rel="noopener noreferer">CSCvm35110</a>
                """.format(used / threshold)

        # return ise_sig.result(borg3.enums.Severity.WARNING)
        print(ise_sig_text)

elif profiler_match:
    ise_sig_issue_hit = True
    ise_sig_text = """
                There is evidence that the PROFILER_PROFILED table on this MnT node is not properly being purged. This eventually could lead to the MnT database filling up. Please refer to the bug
                link below for patched versions of ISE. The MnT data threshold is currently below 75% so there is time to plan out patching this deployment.<br>
                Reference: <a href="https://scripts.cisco.com/app/quicker_cdets/?bug=CSCvm35110" target="_blank" rel="noopener noreferer">CSCvm35110</a>
            """
    ise_sig_external_text = """
                There is evidence that the PROFILER_PROFILED table on this MnT node is not properly being purged. This eventually could lead to the MnT database filling up. Please refer to the bug
                link below for patched versions of ISE. The MnT data threshold is currently below 75% so there is time to plan out patching this deployment.<br>
                Reference: <a href="https://bst.cloudapps.cisco.com/bugsearch/bug/CSCvm35110" target="_blank" rel="noopener noreferer">CSCvm35110</a>
            """
    print(ise_sig_text)


else:
    print("no match")








'''     ise_sig.issue_hit = True

        ise_sig.text = """
                    There is evidence that the PROFILER_PROFILED table on this MnT node is not properly being purged. This eventually could lead to the MnT database filling up. Please refer to the bug
                    link below for patched versions of ISE. The MnT data threshold is currently below 75% so there is time to plan out patching this deployment.<br>
                    Reference: <a href="https://scripts.cisco.com/app/quicker_cdets/?bug=CSCvm35110" target="_blank" rel="noopener noreferer">CSCvm35110</a>
                """
        ise_sig.external_text = """
                    There is evidence that the PROFILER_PROFILED table on this MnT node is not properly being purged. This eventually could lead to the MnT database filling up. Please refer to the bug
                    link below for patched versions of ISE. The MnT data threshold is currently below 75% so there is time to plan out patching this deployment.<br>
                    Reference: <a href="https://bst.cloudapps.cisco.com/bugsearch/bug/CSCvm35110" target="_blank" rel="noopener noreferer">CSCvm35110</a>
                """

        # return ise_sig.result(borg3.enums.Severity.NOTICE)
'''


