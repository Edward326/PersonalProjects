# USER INSTRUCTIONS: DUMMY VERSIONATING SYSTEM WITH THREAT FINDING CONTROL//OS[cite: 3]

## Command Line Usage[cite: 3]
* In the command line, the name of the executable is called to create a process responsible for launching the versioning program[cite: 3].
* The usage syntax involves specifying the executable (for example, `./execProgName`) followed by arguments[cite: 3].
* The arguments represent the directories we want to version[cite: 3].

## Program Execution Results[cite: 3]
The program will return messages directly to the screen, depending on the execution state[cite: 3].

### 1. Program terminated abnormally[cite: 3]
This message appears in the following situations[cite: 3]:
* When the number of directories we want to version is greater than `maxDirToVers(macro)` or less than 1[cite: 3].
* When the initial value for `maxDirToVers` is too large, a situation in which the operating system would no longer have available PIDs to create all the necessary processes through the `fork` function[cite: 3].

### 2. Program terminated succefully[cite: 3]
This message appears if the conditions for abnormal termination are not met[cite: 3].
* The program will iterate through each directory and create a dedicated process for it[cite: 3].
* Within each process, algorithm K will be applied[cite: 3].
* The processes responsible for applying algorithm K will run in parallel[cite: 3].

## Algorithm K[cite: 3]
The algorithm is applied only if the directory received from the command line simultaneously meets the following conditions: it exists in the system, it is a directory-type file, and it has not been entered again from the keyboard[cite: 3].
The algorithm consists of two main stages: cleaning the directory and versioning it[cite: 3].

### Stage 1: Cleaning the directory of malicious files[cite: 3]
* The entire file tree from the specified directory is recursively traversed[cite: 3].
* If the element is a directory, the recursive traversal continues[cite: 3].
* If the element is a file, it is analyzed[cite: 3].
* If the file has permissions other than `000 (u+g+o)`, it is ignored[cite: 3].
* If the file has `000` permissions (which indicates possible malware), a grandchild process (child process) is created that launches the `checkIntegrity.sh` script into execution[cite: 3].
* The `checkIntegrity.sh` script checks three conditions (1, 2, or 3)[cite: 3].
* If at least one of these conditions is true (`1||2||3`), the process writes the message "CORRUPT" into a pipe[cite: 3].
* If none is true, the process writes the message "SUCCES" into the pipe[cite: 3].
* The parent process (the one applying algorithm K on that directory) reads the information from the pipe[cite: 3].
* If the read message is "CORRUPT", the malicious file is moved to the pre-established `isolatedFiles` directory, deleted from the current directory, and the number of malicious files found is counted[cite: 3].
* Each grandchild process that analyzes a file with `000` permissions runs in parallel with the other traversals[cite: 3].
* A different pipe is used for each grandchild process to avoid overwriting data by other processes that detect malicious files[cite: 3].

### Stage 2: Versioning the directory[cite: 3]
* The versioning process takes place only after the malicious files have been found and deleted[cite: 3].
* There is a standard directory named `localSaves` where snapshots are saved[cite: 3].
* In `localSaves`, each directory has its own folder containing binary data (`metadata.bin`)[cite: 3].
* The `metadata.bin` file keeps the structure of the file tree, noting the details from the INode of each element[cite: 3].
* The system checks if the directory has been previously versioned[cite: 3].
* If the directory is NOT versioned, the system versions it[cite: 3].
* If the directory IS already versioned, the system keeps a reference to the old version in `localSaves` and creates a reference to the current directory (the new version)[cite: 3].
* The two references are compared, and if differences are identified, the new version is saved[cite: 3].
* The system can recognize modifications such as: changing the name of the main directory, changing the name of a subdirectory, changing the name of a file, adding data to a file (increased size), or, if the size remained identical, checking the last modification date[cite: 3].
* A pipe is used through which a content is written to standard output, then redirected to the write end of the pipe[cite: 3].
* This content reports the state of the directory: whether it was already versioned and modifications were found or not, or if it was not versioned and has just been versioned[cite: 3].

## Returning the Results[cite: 3]
* Each process that applied algorithm K returns to the parent process the number of malicious files it found[cite: 3].
* The parent process waits for the completion of all versioning processes and sums up the number of malicious files detected by each one[cite: 3].

## Execution Examples[cite: 3]
* Running the `./ex` command will generate a `-1` error and display the `program terminated abnormally` message[cite: 3].
* Running the `./ex test1 test2` command will return `0` (success) and display the status of each directory after versioning (if the directory exists, is of DIR type, and has not been submitted for versioning again)[cite: 3].
* At the end of a successful execution, the program will display the following information[cite: 3]:
    * `program terminated succefully`[cite: 3]
    * `mainProcess terminated with a total of Malitious Files found: [number of malicious files found]`[cite: 3]
    * `totalExecTime: %f sec`[cite: 3]

## Important Note[cite: 3]
* To correctly view a buffer-type file (which contains non-ASCII characters), it is recommended to open it in VSCode[cite: 3].