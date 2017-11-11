all:
	g++ --std=c++11 migration.cpp test-migration.cpp -I. -lhwloc -lnuma -O3 -o ftt -lpthread -g
	g++ --std=c++11 migration.cpp test-migration.cpp -I. -lhwloc -lnuma -O3 -o pmt -DUSE_PAGE_MIGRATION -lpthread -g
debug: 
	g++ --std=c++11 migration.cpp test-migration.cpp -I. -lhwloc -lnuma -o ftt -lpthread -g
	g++ --std=c++11 migration.cpp test-migration.cpp -I. -lhwloc -lnuma -o pmt -DUSE_PAGE_MIGRATION -lpthread -g
