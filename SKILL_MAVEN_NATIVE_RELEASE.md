# Skill: Java Native 库项目 CI/CD 与 Maven Central 发布

## 适用场景
Java 项目包含 JNI native 代码，需要跨平台编译并发布到 Maven Central。

## 前置条件
- GitHub 仓库
- Sonatype Central 账号 (https://central.sonatype.com)
- GPG 密钥

---

## 1. 项目结构

```
project/
├── pom.xml
├── CMakeLists.txt
├── src/main/java/          # Java 代码
├── src/main/native/        # C/C++ 代码
├── src/main/resources/native/
│   ├── linux-x86_64/
│   ├── darwin-aarch64/
│   └── windows-x86_64/
└── .github/workflows/
    ├── build.yml           # CI 构建
    └── release.yml         # 发布到 Maven Central
```

---

## 2. pom.xml 配置

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>com.example</groupId>
    <artifactId>project-name</artifactId>
    <version>1.0.0</version>
    <packaging>jar</packaging>

    <!-- Maven Central 必需的元数据 -->
    <name>project-name</name>
    <url>https://github.com/username/project-name</url>
    <description>Project description</description>

    <licenses>
        <license>
            <name>The Apache License, Version 2.0</name>
            <url>http://www.apache.org/licenses/LICENSE-2.0.txt</url>
        </license>
    </licenses>

    <developers>
        <developer>
            <name>Your Name</name>
            <email>your@email.com</email>
        </developer>
    </developers>

    <scm>
        <connection>scm:git:git://github.com/username/project-name.git</connection>
        <developerConnection>scm:git:ssh://github.com/username/project-name.git</developerConnection>
        <url>https://github.com/username/project-name/tree/main</url>
    </scm>

    <properties>
        <maven.compiler.source>1.8</maven.compiler.source>
        <maven.compiler.target>1.8</maven.compiler.target>
        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
        <gpg.skip>true</gpg.skip>
        <gpg.keyname>YOUR_GPG_KEY_ID</gpg.keyname>
    </properties>

    <build>
        <plugins>
            <!-- 源码 JAR -->
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-source-plugin</artifactId>
                <version>3.3.0</version>
                <executions>
                    <execution>
                        <id>attach-sources</id>
                        <goals><goal>jar-no-fork</goal></goals>
                    </execution>
                </executions>
            </plugin>
            
            <!-- Javadoc JAR -->
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-javadoc-plugin</artifactId>
                <version>3.6.3</version>
                <configuration>
                    <doclint>none</doclint>
                </configuration>
                <executions>
                    <execution>
                        <id>attach-javadocs</id>
                        <goals><goal>jar</goal></goals>
                    </execution>
                </executions>
            </plugin>
            
            <!-- GPG 签名 -->
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-gpg-plugin</artifactId>
                <version>3.1.0</version>
                <executions>
                    <execution>
                        <id>sign-artifacts</id>
                        <phase>verify</phase>
                        <goals><goal>sign</goal></goals>
                        <configuration>
                            <keyname>${gpg.keyname}</keyname>
                            <passphraseServerId>${gpg.keyname}</passphraseServerId>
                        </configuration>
                    </execution>
                </executions>
            </plugin>
            
            <!-- Sonatype Central 发布 -->
            <plugin>
                <groupId>org.sonatype.central</groupId>
                <artifactId>central-publishing-maven-plugin</artifactId>
                <version>0.4.0</version>
                <extensions>true</extensions>
                <configuration>
                    <publishingServerId>central</publishingServerId>
                    <tokenAuth>true</tokenAuth>
                </configuration>
            </plugin>
        </plugins>
    </build>
</project>
```

---

## 3. CI 构建 (.github/workflows/build.yml)

```yaml
name: Build

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build-native:
    strategy:
      fail-fast: false
      matrix:
        include:
          - os: ubuntu-latest
            platform: linux-x86_64
          - os: macos-14
            platform: darwin-aarch64
          - os: windows-latest
            platform: windows-x86_64
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-java@v4
        with:
          distribution: temurin
          java-version: '11'
      - name: Build native
        run: |
          cmake -B build -DCMAKE_BUILD_TYPE=Release
          cmake --build build --config Release
      - uses: actions/upload-artifact@v4
        with:
          name: native-${{ matrix.platform }}
          path: |
            build/*.so
            build/*.dylib
            build/*.dll
            build/Release/*.dll

  package:
    needs: build-native
    runs-on: ubuntu-latest
    strategy:
      matrix:
        java: ['8', '11', '17', '21']
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-java@v4
        with:
          distribution: temurin
          java-version: ${{ matrix.java }}
      - uses: actions/download-artifact@v4
        with:
          path: natives
      - name: Copy natives
        run: |
          find natives -name "*.so" -exec cp {} src/main/resources/native/linux-x86_64/ \;
          find natives -name "*.dylib" -exec cp {} src/main/resources/native/darwin-aarch64/ \;
          find natives -name "*.dll" -exec cp {} src/main/resources/native/windows-x86_64/ \;
      - name: Build and test
        run: mvn -B verify
      - name: Upload JAR
        if: matrix.java == '8'
        uses: actions/upload-artifact@v4
        with:
          name: project-jar
          path: target/*.jar
```

---

## 4. 发布 Workflow (.github/workflows/release.yml)

```yaml
name: Release

on:
  workflow_dispatch:
    inputs:
      version:
        description: 'Release version (e.g., 1.0.0)'
        required: true

jobs:
  build-native:
    strategy:
      matrix:
        include:
          - os: ubuntu-latest
            platform: linux-x86_64
          - os: macos-14
            platform: darwin-aarch64
          - os: windows-latest
            platform: windows-x86_64
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-java@v4
        with:
          distribution: temurin
          java-version: '11'
      - name: Build native
        run: |
          cmake -B build -DCMAKE_BUILD_TYPE=Release
          cmake --build build --config Release
      - uses: actions/upload-artifact@v4
        with:
          name: native-${{ matrix.platform }}
          path: |
            build/*.so
            build/*.dylib
            build/*.dll
            build/Release/*.dll

  release:
    needs: build-native
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-java@v4
        with:
          distribution: temurin
          java-version: '8'
          server-id: central
          server-username: MAVEN_USERNAME
          server-password: MAVEN_PASSWORD
      - name: Import GPG key
        run: |
          echo "${{ secrets.GPG_PRIVATE_KEY }}" | gpg --batch --import
          gpg --list-secret-keys --keyid-format LONG
      - uses: actions/download-artifact@v4
        with:
          path: natives
      - name: Copy natives
        run: |
          find natives -name "*.so" -exec cp {} src/main/resources/native/linux-x86_64/ \;
          find natives -name "*.dylib" -exec cp {} src/main/resources/native/darwin-aarch64/ \;
          find natives -name "*.dll" -exec cp {} src/main/resources/native/windows-x86_64/ \;
      - name: Set version
        run: mvn versions:set -DnewVersion=${{ github.event.inputs.version }}
      - name: Publish to Maven Central
        run: |
          mvn -B deploy -Dgpg.skip=false \
            -Dgpg.passphrase="${{ secrets.GPG_PASSPHRASE }}" \
            -Dgpg.executable=gpg
        env:
          MAVEN_USERNAME: ${{ secrets.MAVEN_CENTRAL_USERNAME }}
          MAVEN_PASSWORD: ${{ secrets.MAVEN_CENTRAL_PASSWORD }}
```

---

## 5. GitHub Secrets 配置

在仓库 Settings → Secrets and variables → Actions 添加：

| Secret | 说明 | 获取方式 |
|--------|------|----------|
| `MAVEN_CENTRAL_USERNAME` | Sonatype 用户名 | central.sonatype.com → View Account → Generate User Token |
| `MAVEN_CENTRAL_PASSWORD` | Sonatype 密码 | 同上 |
| `GPG_PRIVATE_KEY` | GPG 私钥 | `gpg --armor --export-secret-keys KEY_ID` |
| `GPG_PASSPHRASE` | GPG 密码 | 创建密钥时设置的密码 |

---

## 6. 发布流程

1. 在 GitHub Actions 页面选择 "Release" workflow
2. 点击 "Run workflow"
3. 输入版本号（如 `1.0.0`）
4. 等待构建完成
5. 在 https://central.sonatype.com 查看发布状态

---

## 常见问题

### macOS x86_64 runner 不稳定
- 使用 `macos-14` (ARM) 替代，Intel Mac 用户可通过 Rosetta 2 运行

### Javadoc 报错 HTML 实体
- 添加 `<doclint>none</doclint>` 配置

### GPG 签名失败 "No secret key"
- 确保 `GPG_PRIVATE_KEY` 包含完整的 `-----BEGIN/END PGP PRIVATE KEY BLOCK-----`
- 检查 `gpg --list-secret-keys` 输出的密钥 ID 与 pom.xml 中的 `gpg.keyname` 一致

### JDK 8 在 macOS ARM 不可用
- Native 构建使用 JDK 11（JNI ABI 兼容）
- 打包和发布使用 JDK 8
