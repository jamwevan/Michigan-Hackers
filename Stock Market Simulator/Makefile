SHELL = /bin/bash
UNIQNAME    = jamwevan

IDENTIFIER  = 0E04A31E0D60C01986ACB20081C9D8722A1899B6

EXECUTABLE  = market

PROJECTFILE = $(or $(wildcard project*.cpp $(EXECUTABLE).cpp), main.cpp)

REMOTE_PATH := eecs281_$(EXECUTABLE)_sync

PATH := /usr/um/gcc-11.3.0/bin:$(PATH)
LD_LIBRARY_PATH := /usr/um/gcc-11.3.0/lib64
LD_RUN_PATH := /usr/um/gcc-11.3.0/lib64


.SUFFIXES:

CXX         = g++


%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $*.cpp

TESTSOURCES = $(wildcard test*.cpp)
TESTSOURCES := $(filter-out $(PROJECTFILE),$(TESTSOURCES))

SOURCES     = $(wildcard *.cpp)
SOURCES     := $(filter-out $(TESTSOURCES), $(SOURCES))
OBJECTS     = $(SOURCES:%.cpp=%.o)

CXXFLAGS = -std=c++17 -Wconversion -Wall -Werror -Wextra -pedantic

debug: CXXFLAGS += -g3 -DDEBUG -fsanitize=address -fsanitize=undefined -D_GLIBCXX_DEBUG
debug:
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(EXECUTABLE)_debug
.PHONY: debug

release: CXXFLAGS += -O3 -DNDEBUG
release: $(EXECUTABLE)
.PHONY: release

valgrind: CXXFLAGS += -g3
valgrind:
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(EXECUTABLE)_valgrind
.PHONY: valgrind


profile: CXXFLAGS += -g3
profile:
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(EXECUTABLE)_profile
.PHONY: profile

static:
	cppcheck --enable=all --suppress=missingIncludeSystem \
      $(SOURCES) *.h *.hpp
.PHONY: static

FULL_SUBMITFILE = fullsubmit.tar.gz
PARTIAL_SUBMITFILE = partialsubmit.tar.gz
UNGRADED_SUBMITFILE = ungraded.tar.gz

NO_IDENTIFIER = xcode_redirect.hpp,getopt.h,getopt.c,xgetopt.h

identifier: $(foreach tsrc,$(wildcard test*.cpp),$(eval NO_IDENTIFIER := $(NO_IDENTIFIER),$(tsrc)))
identifier:
	@if [ $$(grep --include=*.{h,hpp,c,cpp} --exclude={$(NO_IDENTIFIER)} --directories=skip -L $(IDENTIFIER) * | wc -l) -ne 0 ]; then \
		printf "Missing project identifier in file(s): "; \
		echo `grep --include=*.{h,hpp,c,cpp} --exclude={$(NO_IDENTIFIER)} --directories=skip -L $(IDENTIFIER) *`; \
		exit 1; \
	else \
		rm -f $(PARTIAL_SUBMITFILE) $(FULL_SUBMITFILE) $(UNGRADED_SUBMITFILE); \
	fi
.PHONY: identifier

all: debug release
all: profile valgrind
.PHONY: all

$(EXECUTABLE): $(OBJECTS)
ifneq ($(EXECUTABLE), executable)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $(EXECUTABLE)
else
	@echo Edit EXECUTABLE variable in Makefile.
	@echo Using default a.out.
	$(CXX) $(CXXFLAGS) $(OBJECTS)
endif

TESTS       = $(TESTSOURCES:%.cpp=%)

define make_tests
    ifeq ($$(PROJECTFILE),)
	    @echo Edit PROJECTFILE variable to .cpp file with main\(\)
	    @exit 1
    endif
    SRCS = $$(filter-out $$(PROJECTFILE), $$(SOURCES))
    OBJS = $$(SRCS:%.cpp=%.o)
    HDRS = $$(wildcard *.h *.hpp)
    $(1): CXXFLAGS += -g3 -DDEBUG
    $(1): $$(OBJS) $$(HDRS) $(1).cpp
	$$(CXX) $$(CXXFLAGS) $$(OBJS) $(1).cpp -o $(1)
endef
$(foreach test, $(TESTS), $(eval $(call make_tests, $(test))))

alltests: $(TESTS)
.PHONY: alltests

clean:
	rm -Rf *.dSYM
	rm -f $(OBJECTS) $(EXECUTABLE) $(EXECUTABLE)_debug
	rm -f $(EXECUTABLE)_valgrind $(EXECUTABLE)_profile $(TESTS) perf.data* \
      $(PARTIAL_SUBMITFILE) $(FULL_SUBMITFILE) $(UNGRADED_SUBMITFILE)
.PHONY: clean

EXCLUDE_FILES = getopt.\?

FULL_SUBMITFILES=$(filter-out $(wildcard test*.cpp), \
                   $(wildcard Makefile *.h *.hpp *.cpp test*.txt))

$(FULL_SUBMITFILE): $(FULL_SUBMITFILES)
	rm -f $(PARTIAL_SUBMITFILE) $(FULL_SUBMITFILE) $(UNGRADED_SUBMITFILE)
	COPYFILE_DISABLE=true tar --exclude=$(EXCLUDE_FILES) -vczf $(FULL_SUBMITFILE) $(FULL_SUBMITFILES)
	@echo !!! Final submission prepared, test files included... READY FOR GRADING !!!

PARTIAL_SUBMITFILES=$(filter-out $(wildcard test*.txt), $(FULL_SUBMITFILES))
$(PARTIAL_SUBMITFILE): $(PARTIAL_SUBMITFILES)
	rm -f $(PARTIAL_SUBMITFILE) $(FULL_SUBMITFILE) $(UNGRADED_SUBMITFILE)
	COPYFILE_DISABLE=true tar --exclude=$(EXCLUDE_FILES) -vczf $(PARTIAL_SUBMITFILE) \
      $(PARTIAL_SUBMITFILES)
	@echo !!! WARNING: No test files included. Use 'make fullsubmit' to include test files. !!!

UNGRADED_SUBMITFILES=$(filter-out Makefile, $(PARTIAL_SUBMITFILES))
$(UNGRADED_SUBMITFILE): $(UNGRADED_SUBMITFILES)
	rm -f $(PARTIAL_SUBMITFILE) $(FULL_SUBMITFILE) $(UNGRADED_SUBMITFILE)
	@touch __ungraded
	COPYFILE_DISABLE=true tar --exclude=$(EXCLUDE_FILES) -vczf $(UNGRADED_SUBMITFILE) \
      $(UNGRADED_SUBMITFILES) __ungraded
	@rm -f __ungraded
	@echo !!! WARNING: This submission will not be graded. !!!

fullsubmit: identifier $(FULL_SUBMITFILE)
partialsubmit: identifier $(PARTIAL_SUBMITFILE)
ungraded: identifier $(UNGRADED_SUBMITFILE)
.PHONY: fullsubmit partialsubmit ungraded

sync2caen:
ifeq ($(UNIQNAME), youruniqname)
	@echo Edit UNIQNAME variable in Makefile.
	@exit 1;
endif
	rsync \
      -av \
      --delete \
      --exclude '*.o' \
      --exclude '$(EXECUTABLE)' \
      --exclude '$(EXECUTABLE)_debug' \
      --exclude '$(EXECUTABLE)_valgrind' \
      --exclude '$(EXECUTABLE)_profile' \
      --exclude '.git*' \
      --exclude '.vs*' \
      --exclude '*.code-workspace' \
      --filter=":- .gitignore" \
      "."/ \
      "$(UNIQNAME)@login.engin.umich.edu:$(REMOTE_PATH)/"
	echo "Files synced to CAEN at ~/$(REMOTE_PATH)/"
.PHONY: sync2caen

define MAKEFILE_HELP
EECS281 Advanced Makefile Help
* This Makefile uses advanced techniques, for more information:
    $$ man make

* General usage
    1. Follow directions at each "TODO" in this file.
       a. Set EXECUTABLE equal to the name from the project specification.
       b. Set PROJECTFILE equal to the name of the source file with main()
       c. Add any dependency rules specific to your files.
    2. Build, test, submit... repeat as necessary.

* Preparing submissions
    A) To build 'partialsubmit.tar.gz', a tarball without tests used to
       find buggy solutions in the autograder.

           *** USE THIS ONLY FOR TESTING YOUR SOLUTION! ***

       This is useful for faster autograder runs during development and
       free submissions if the project does not build.
           $$ make partialsubmit
    B) Build 'fullsubmit.tar.gz' a tarball complete with autograder test
       files.

           *** ALWAYS USE THIS FOR FINAL GRADING! ***

       It is also useful when trying to find buggy solutions in the
       autograder.
           $$ make fullsubmit
    C) Build 'ungraded.tar.gz' a tarball without tests that the autograder
       will not try to build.

           *** USE THIS ONLY TO UPLOAD CODE TO THE AUTOGRADER FOR STAFF ***

       This is only useful for a staff member who wants to view your most
       recent code on the autograder.
           $$ make ungraded

* Unit testing support
    A) Source files for unit testing should be named test*.cpp.  Examples
       include test_input.cpp or test3.cpp.
    B) Automatic build rules are generated to support the following:
           $$ make test_input
           $$ make test3
           $$ make alltests        (this builds all test drivers)
    C) If test drivers need special dependencies, they must be added
       manually.
    D) IMPORTANT: NO SOURCE FILES WITH NAMES THAT BEGIN WITH test WILL BE
       ADDED TO ANY SUBMISSION TARBALLS.

* Static Analysis support
    A) Matches current autograder style grading tests
    B) Usage:
           $$ make static

* Sync to CAEN support
    A) Requires an .ssh/config file with a login.engin.umich.edu host
       defined, SSH Multiplexing enabled, and an open SSH connection.
    B) Edit the REMOTE_BASEDIR variable if default is not preferred.
    C) Usage:
           $$ make sync2caen
endef
export MAKEFILE_HELP

help:
	@echo "$$MAKEFILE_HELP"
.PHONY: help

