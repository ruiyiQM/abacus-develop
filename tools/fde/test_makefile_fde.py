import re
import subprocess
import shutil
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'source'


class MakefileFdeSourcesTest(unittest.TestCase):
    def setUp(self):
        self.make = shutil.which("make")
        self.assertIsNotNone(self.make, "GNU make is required for the legacy FDE build regression")

    def test_cmake_production_sources_are_resolved_by_legacy_make(self):
        cmake = (SOURCE / 'source_lcao/module_fde/CMakeLists.txt').read_text()
        names = re.findall(r'\$\{CMAKE_CURRENT_SOURCE_DIR\}/([^\s)]+\.cpp)', cmake)
        self.assertTrue(names, 'CMake FDE source list was not found')
        expected = [SOURCE / 'source_lcao/module_fde' / name for name in names]
        expected.append(SOURCE / 'source_io/module_parameter/read_input_item_fde.cpp')
        makefile = 'include Makefile.Objects\nprint:\n\t@echo $(OBJS_ABACUS)\n\t@echo $(VPATH)\n'
        result = subprocess.run([self.make, '-s', '-f', '-', 'print'], input=makefile,
                                text=True, cwd=SOURCE, capture_output=True, check=True)
        objects, vpath = result.stdout.splitlines()
        objects = objects.split()
        directories = [SOURCE / path.strip() for path in vpath.split(':')]
        for source in expected:
            with self.subTest(source=source):
                obj = source.stem + '.o'
                if source.name == 'read_input_item_fde.cpp':
                    obj = 'module_parameter/' + obj
                self.assertEqual(objects.count(obj), 1, f'{obj} must appear once in OBJS_ABACUS')
                relative_source = Path(obj).with_suffix('.cpp')
                matches = [directory / relative_source for directory in directories
                           if (directory / relative_source).is_file()]
                self.assertTrue(matches, f'{relative_source} is not found via VPATH')
                self.assertEqual(matches[0].resolve(), source.resolve(),
                                 f'{relative_source} resolves to the wrong source: {matches}')

    def test_libxc_switch_enables_provider(self):
        makefile = 'print-flags:\n\t@echo $(HONG)\n'
        result = subprocess.run([self.make, '-s', '-f', 'Makefile', '-f', '-',
                                 'LIBXC_DIR=/tmp/libxc', 'print-flags'], input=makefile,
                                text=True, cwd=SOURCE, capture_output=True, check=True)
        self.assertIn('-D__LIBXC', result.stdout.split())


if __name__ == '__main__':
    unittest.main()
